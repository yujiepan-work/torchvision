import datetime
import heapq
import os
import random
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict


def avail_cuda_list(memory_requirement):
    p = subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A5 GPU | grep Free",
        shell=True,
        stdout=subprocess.PIPE,
    )
    free_mem = [(-int(x.split()[2]), i) for i, x in enumerate(p.stdout.readlines())]
    heapq.heapify(free_mem)

    def get_one():
        nonlocal free_mem
        free, i = free_mem[0]
        if free + memory_requirement > -20:
            return -1
        heapq.heapreplace(free_mem, (free + memory_requirement, i))
        return i

    result = []
    i = get_one()
    while i >= 0:
        result.append(i)
        i = get_one()

    return result


def main():
    n_card = str(sys.argv[-1])
    assert n_card.isdigit()
    n_card = int(n_card)

    while True:
        cards = avail_cuda_list(28500)
        if len(set(cards)) >= n_card:
            break
        print("Available cards:", cards, ", waiting...")
        time.sleep(60)


main()
