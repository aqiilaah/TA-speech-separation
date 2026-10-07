"""Parse the multiscale settings shared by the wizard and training scripts."""

import os
import re


def parse_multiscale_settings(kernels="16/32/64", strides="8/8/8"):
    if not re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*/[1-9][0-9]*", kernels):
        raise ValueError("Kernel sizes must be three positive integers, e.g. 16/32/64")
    if not re.fullmatch(r"[1-9][0-9]*(/[1-9][0-9]*/[1-9][0-9]*)?", strides):
        raise ValueError("Strides must be one or three positive integers, e.g. 8 or 8/16/32")
    kernels = tuple(map(int, kernels.split("/")))
    strides = tuple(map(int, strides.split("/")))
    if len(strides) == 1:
        strides *= 3
    if any(a >= b for a, b in zip(kernels, kernels[1:])):
        raise ValueError("Kernel sizes must increase from short to long")
    if any(s > k for k, s in zip(kernels, strides)):
        raise ValueError("Each stride must be no larger than its matching kernel")
    return kernels, strides


def apply_multiscale_environment(config):
    kernels, strides = parse_multiscale_settings(
        os.environ.get("TSS_MULTISCALE_KERNELS", "16/32/64"),
        os.environ.get("TSS_MULTISCALE_STRIDES", "8/8/8"),
    )
    for name in ("encoder", "decoder"):
        config[name]["kernel_sizes"] = kernels
        config[name]["stride"] = strides[0] if len(set(strides)) == 1 else strides


if __name__ == "__main__":
    import sys

    try:
        kernels, strides = parse_multiscale_settings(*sys.argv[1:])
    except ValueError as exc:
        raise SystemExit(str(exc))
    print("/".join(map(str, kernels)), "/".join(map(str, strides)))
