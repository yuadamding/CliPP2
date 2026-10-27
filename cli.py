from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_DEVICE, DEFAULT_MAX_CLUSTERS, DEFAULT_MAX_MAJOR_CN, FitConfig, resolve_fit_config
from ._version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipp2", allow_abbrev=False,
        description="Fit the free-center tree-proposed joint-region multiplicity-mixture estimator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", allow_abbrev=False,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inputs = fit.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-file", help="Canonical tumor TSV (optionally gzip-compressed).")
    inputs.add_argument("--input-dir", dest="input_file", metavar="INPUT_DIR",
                        help="Tumor directory with regional snv.txt/cna.txt/purity.txt bundles.")
    fit.add_argument("--outdir", default="clipp2_results")
    fit.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE)
    fit.add_argument(
        "--max-major-cn", type=int, default=DEFAULT_MAX_MAJOR_CN,
        help="Exclude a mutation only from regions where any CN state's major CN exceeds this limit.",
    )
    fit.add_argument("--verbose", action="store_true")
    fit.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_CLUSTERS,
                     help="Largest occupied cluster capacity (positive integer; no fixed upper limit).")
    fit.add_argument("--grid-cache-mb", type=int, default=64,
                     help="Host initialization-grid cache budget in MiB; 0 streams without caching.")
    fit.add_argument("--device-cache-mb", type=int, default=256,
                     help="Compatibility option; the native backend does not retain a device cache.")
    fit.add_argument("--workspace-mb", type=int, default=64,
                     help="Native CPU/CUDA scratch budget in MiB (positive; not total memory).")
    fit.add_argument("--native-host-memory-mb", type=int, default=None,
                     help="Tracked native host memory cap in MiB; auto uses 75%% of available headroom.")
    fit.add_argument("--device-memory-mb", type=int, default=None,
                     help="Tracked device memory cap in MiB; auto uses 80%% of free device headroom.")
    fit.add_argument("--cpu-threads", type=int, default=None,
                     help="CPU execution budget; fallback CLIPP2_CPU_THREADS, OMP_NUM_THREADS, then affinity/quota.")
    fit.add_argument("--version", action="version", version=__version__)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _fit_config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def _fit_config_from_args(args: argparse.Namespace) -> FitConfig:
    return resolve_fit_config(device=args.device, verbose=args.verbose,
                              max_major_cn=args.max_major_cn, max_clusters=args.max_clusters,
                              grid_cache_mb=args.grid_cache_mb, device_cache_mb=args.device_cache_mb,
                              workspace_mb=args.workspace_mb,
                              native_host_memory_mb=args.native_host_memory_mb,
                              device_memory_mb=args.device_memory_mb, cpu_threads=args.cpu_threads)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from .api import process_tumor

    process_tumor(
        tumor_file=Path(args.input_file),
        outdir=Path(args.outdir),
        fit_config=_fit_config_from_args(args),
    )


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
