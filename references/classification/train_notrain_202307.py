import datetime
import os
import time
import warnings
import json

import presets
import torch
import torch.utils.data
import torch.nn.functional as F
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

from copy import deepcopy

EVAL_BEFORE_TRAIN = 1


def load_state_dict_and_manual_mask(model: torch.nn.Module, sd: dict):
    from state_dict_patch import resolve_structured_mask, calc_sparsity

    new_sd, preserved_by_layer = resolve_structured_mask(sd)
    model.load_state_dict(new_sd, strict=True)
    return model, preserved_by_layer, calc_sparsity(preserved_by_layer)


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    if model is None:
        return 0, 0, 0
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
        # 2023 patch: does not use movement, but will prune manually
        nncf_config["log_dir"] = args.output_dir
        if utils.is_main_process():
            shutil.copy(args.nncf_config, args.output_dir)

        if (args.manual_load is not None or args.max_steps > 0) and "compression" in nncf_config:
            # save the calibration time for manual load
            override_qcfg_init = dict(
                range=dict(num_init_samples=0), batchnorm_adaptation=dict(num_bn_adaptation_samples=0)
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

        SUMMARY = {}
        if args.manual_load is not None:
            sd = torch.load(args.manual_load, map_location="cpu")["model"]
            model, preserved_by_layer, linear_sparsity = load_state_dict_and_manual_mask(model, sd)
            print(f"Loaded model state from: {args.manual_load}")
            SUMMARY["model_path"] = args.manual_load
            SUMMARY["structure_mask"] = preserved_by_layer
            SUMMARY["linear_sparsity"] = linear_sparsity

            # the bianry masks are finalized. So simply freeze them
            if hasattr(compression_ctrl, "child_ctrls"):
                mvmt_ctrl = compression_ctrl.child_ctrls[0]  # TODO: index 0, hard coded.
            else:
                mvmt_ctrl = compression_ctrl
            mvmt_ctrl.scheduler._disable_importance_grad()
            # mvmt_ctrl.reset_independent_structured_mask()
            # mvmt_ctrl.resolve_structured_mask()
            # mvmt_ctrl.populate_structured_mask()
            
            # do actually export
            compression_ctrl.export_model(os.path.join(args.output_dir, 'model.onnx'))
            return
        # else:
        #     torch.save({'model': model.state_dict()}, '/home/yujiepan/work2/jpqd-vit/LOGS/ptq_model/jpq-pytorch.v3-512.bin')
        print("\n".join(sorted(model.state_dict().keys())))

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # TODO: nncf with syncBN

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        # TODO: find_unused_parameters will cause slow training.
        model_without_ddp = model.module
        if compression_ctrl:
            compression_ctrl.distributed()
    print(model)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Test model accuracy before the training starts.")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    result = evaluate(model, criterion, data_loader_test, device=device, print_freq=args.print_freq)
    SUMMARY["top1"] = result[0]
    SUMMARY["top5"] = result[1]
    SUMMARY["loss"] = result[2]
    ss = deepcopy(SUMMARY["structure_mask"])
    del SUMMARY["structure_mask"]  # make it at end
    SUMMARY['num_preserved_heads_channels_summary'] = {k: (len(v[0]), len(v[1])) for k, v in ss.items()}
    SUMMARY["structure_mask"] = ss
    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(SUMMARY, f, indent=2)
    return


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
    parser.add_argument("--teacher", default=None, type=str, help="teacher model name")
    parser.add_argument("--teacher_weights", default=None, type=str, help="the weights enum name to load for teacher")
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
