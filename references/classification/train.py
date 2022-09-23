import datetime
import os
import time
import warnings

import presets
import torch
import torch.utils.data
import torchvision
import transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

import shutil
from nncf import NNCFConfig
from nncf.torch import register_default_init_args
from nncf.torch import create_compressed_model

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

EVAL_BEFORE_TRAIN = 0


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
    log_wandb=False,
    compression_ctrl=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.3e}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value:.0f}"))
    header = f"Epoch: [{epoch}]"

    if compression_ctrl:
        compression_ctrl.scheduler.epoch_step()

    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header, log_wandb)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        if compression_ctrl:
            compression_ctrl.scheduler.step()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss_main = criterion(output, target)
            loss_dict = dict(loss_main=loss_main)
            if compression_ctrl:
                if not hasattr(compression_ctrl, "child_ctrls"):
                    loss_dict["loss_compress"] = compression_ctrl.loss()
                else:
                    for child_ctrl in compression_ctrl.child_ctrls:
                        loss_dict["loss_" + child_ctrl.name] = child_ctrl.loss()
            loss = sum(loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"]
        )  # TODO: first lr group is about importance score if global-lr is false
        if compression_ctrl:
            for loss_name, loss_value in loss_dict.items():
                metric_logger.meters[loss_name].update(loss_value.item())
        movement_ctrl_statistics = compression_ctrl.statistics().movement_sparsity
        metric_logger.update(
            importance_regularization_factor=movement_ctrl_statistics.importance_regularization_factor,
            importance_threshold=movement_ctrl_statistics.importance_threshold,
            relative_sparsity=movement_ctrl_statistics.model_statistics.sparsity_level_for_layers,
        )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    # return 0.5, 0.9, 1.8
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    # gather the stats from all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, metric_logger.loss.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    if args.max_steps > 0:
        n_gpu = (
            torch.distributed.get_world_size() if args.distributed else max(1, torch.cuda.device_count())
        )  # TODO: not tested
        subset_length = args.max_steps * args.batch_size * n_gpu
        print(
            f"Reduce dataset size for train: {len(dataset)}->{subset_length}, val: {len(dataset_test)}->{subset_length}."
        )
        dataset.__class__.__len__ = lambda _: subset_length
        dataset_test.__class__.__len__ = lambda _: subset_length

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    log_wandb = False
    if utils.is_main_process() and args.test_only is False:
        if has_wandb is True and args.wandb_id is not None:
            wandb.init(project=os.getenv("WANDB_PROJECT", "torchvision-train"), name=args.wandb_id, config=args)
            log_wandb = True

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    model = torchvision.models.__dict__[args.model](weights=args.weights, num_classes=num_classes)
    model.to(device)

    compression_ctrl = None
    if args.nncf_config is not None:
        nncf_config = NNCFConfig.from_json(args.nncf_config)
        nncf_config["log_dir"] = args.output_dir
        if utils.is_main_process():
            shutil.copy(args.nncf_config, args.output_dir)

        if (args.manual_load is not None or args.max_steps > 0) and "compression" in nncf_config:
            # save the calibration time for manual load
            override_qcfg_init = dict(
                range=dict(num_init_samples=32), batchnorm_adaptation=dict(num_bn_adaptation_samples=32)
            )
            override_pcfg_param = dict(steps_per_epoch=args.max_steps)
            if isinstance(nncf_config["compression"], list):
                for algo in nncf_config["compression"]:
                    if algo["algorithm"] == "quantization":
                        algo["initializer"].update(override_qcfg_init)
                    if algo["algorithm"] == "movement_sparsity" and args.max_steps > 0:
                        algo["params"].update(override_pcfg_param)
            elif nncf_config["compression"]["algorithm"] == "quantization":
                nncf_config["compression"]["initializer"].update(override_qcfg_init)
            elif nncf_config["compression"]["algorithm"] == "movement_sparsity" and args.max_steps > 0:
                nncf_config["compression"]["params"].update(override_pcfg_param)

        nncf_config = register_default_init_args(
            nncf_config=nncf_config, train_loader=data_loader, device=device
        )  # TODO: distributed_callbacks and execution_parameters
        print(nncf_config)
        compression_ctrl, model = create_compressed_model(model, nncf_config)

        if args.manual_load is not None:
            model.load_state_dict(torch.load(args.manual_load, map_location="cpu")["model"])
            print(f"Loaded model state from: {args.manual_load}")
        # else:
        #     torch.save({'model': model.state_dict()}, '/home/yujiepan/work2/jpqd-vit/LOGS/ptq_model/jpq-pytorch.v3-512.bin')
        print("\n".join(sorted(model.state_dict().keys())))

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # TODO: nncf with syncBN

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    if int(os.environ.get("YUJIE_GLOBAL_LR", "1")) != 1:
        assert len(parameters) == 1, "Currently we assume a global weight decay value."
        importance_params = []
        other_params = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "importance" in name:
                importance_params.append(parameter)
            else:
                other_params.append(parameter)
        print(f"{len(importance_params)} importance params, {len(other_params)} other params.")
        parameters = [{"params": importance_params, "lr": 0.001}, {"params": other_params}]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        # TODO: find_unused_parameters will cause slow training.
        model_without_ddp = model.module
        if compression_ctrl:
            compression_ctrl.distributed()
    print(model)

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(
                model_ema, criterion, data_loader_test, device=device, print_freq=args.print_freq, log_suffix="EMA"
            )
        else:
            evaluate(model, criterion, data_loader_test, print_freq=args.print_freq, device=device)
        return

    print("Start training")
    start_time = time.time()
    if EVAL_BEFORE_TRAIN:
        print("Test model accuracy before the training starts.")
        evaluate(model, criterion, data_loader_test, device=device, print_freq=args.print_freq)

    step_per_epoch = len(data_loader)
    hasfilled = False
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=data_loader,
            device=device,
            epoch=epoch,
            args=args,
            model_ema=model_ema,
            scaler=scaler,
            log_wandb=log_wandb,
            compression_ctrl=compression_ctrl,
        )
        lr_scheduler.step()
        eval_acc1, eval_acc5, eval_loss = evaluate(
            model, criterion, data_loader_test, device=device, print_freq=args.print_freq
        )

        if log_wandb is True:
            global_step = (epoch + 1) * step_per_epoch
            wandb_dict = {
                "eval/global_step": global_step,
                "eval/end_of_epoch": epoch,
                "eval/top1": eval_acc1,
                "eval/top5": eval_acc5,
                "eval/loss": eval_loss,
            }
            wandb.log(data=wandb_dict, step=global_step)

        if model_ema:
            ema_acc1, ema_acc5, ema_loss = evaluate(
                model_ema, criterion, data_loader_test, device=device, print_freq=args.print_freq, log_suffix="EMA"
            )
            if log_wandb is True:
                global_step = (epoch + 1) * step_per_epoch
                wandb_dict = {
                    "ema/global_step": global_step,
                    "ema/end_of_epoch": epoch,
                    "ema/top1": ema_acc1,
                    "ema/top5": ema_acc5,
                    "ema/loss": ema_loss,
                }
                wandb.log(data=wandb_dict, step=global_step)

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
            if compression_ctrl is not None:  # and utils.is_main_process():
                compression_ctrl.export_model(os.path.join(args.output_dir, f"model_{epoch}_{utils.get_rank()}.onnx"))

        if hasfilled is False:
            if hasattr(compression_ctrl, "child_ctrls"):
                mvmt_ctrl = compression_ctrl.child_ctrls[0]  # TODO: index 0
            else:
                mvmt_ctrl = compression_ctrl

            if mvmt_ctrl.__class__.__name__ == "MovementSparsityController":
                if mvmt_ctrl.scheduler.current_epoch + 1 >= mvmt_ctrl.scheduler.warmup_end_epoch:
                    # mvmt_ctrl.report_structured_sparsity(os.path.join(self.args.output_dir, "1-pre"))
                    print("reset_independent_structured_mask")
                    mvmt_ctrl.reset_independent_structured_mask()
                    # mvmt_ctrl.report_structured_sparsity(os.path.join(self.args.output_dir, "2-reset"))
                    print("resolve_structured_mask")
                    mvmt_ctrl.resolve_structured_mask()
                    pth = os.path.join(args.output_dir, "3-resolve")
                    os.makedirs(pth, exist_ok=True)
                    mvmt_ctrl.report_structured_sparsity(pth)
                    utils.save_on_master({"model": model.state_dict()}, os.path.join(pth, "model_resolve.pth"))

                    print("populate_structured_mask")
                    mvmt_ctrl.populate_structured_mask()
                    pth = os.path.join(args.output_dir, "4-pop")
                    os.makedirs(pth, exist_ok=True)
                    mvmt_ctrl.report_structured_sparsity(pth)
                    utils.save_on_master({"model": model.state_dict()}, os.path.join(pth, "model_pop.pth"))
                    compression_ctrl.export_model(os.path.join(pth, f"model_pop_{utils.get_rank()}.onnx"))
                    hasfilled = True

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--wandb_id", default=None, type=str, help="run identifier for wandb dashboard")
    parser.add_argument(
        "--max_steps", default=-1, type=int, help="max number of steps per epoch, -1 means disabled (default -1)"
    )
    parser.add_argument("--nncf_config", default=None, type=str, help="path to nncf config json file")
    parser.add_argument("--manual_load", default=None, type=str, help="path to state dict of nncf wrapped model")
    return parser


if __name__ == "__main__":
    # TODO: a very bad but simple idea to log the timestamp, should delete at final codes
    if 1:
        _print = __builtins__.print

        def print(*args, **kwargs):
            _print(datetime.datetime.now().strftime("%m-%d %H:%M:%S"), *args, **kwargs)

        __builtins__.print = print

    args = get_args_parser().parse_args()
    main(args)
