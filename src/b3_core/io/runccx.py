#!/usr/bin/env python3

import argparse
import multiprocessing
import os
import shutil
import subprocess


def run_single(inp):
    basename = inp.replace(".inp", "")
    frd = f"{basename}.frd"
    if os.path.isfile(frd):
        os.remove(frd)
    assert shutil.which("ccx") is not None
    with open(os.devnull, "w") as devnull:
        subprocess.call(["ccx", basename], stdout=devnull, stderr=devnull)
    if os.path.isfile(frd):
        return frd
    return None


def runccx(inputs):
    for i in inputs:
        assert os.path.isfile(i)
        assert i.endswith(".inp")
    p = multiprocessing.Pool()
    frds = p.map(run_single, inputs)
    return frds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    args = p.parse_args()
    outputfiles = runccx(args.inputs)
    for i in outputfiles:
        print(f"Output in {i}")


if __name__ == "__main__":
    main()
