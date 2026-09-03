"""Thin command-line facade for one canonical CliPP2 fit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .config import (
    COMPUTATION_PROFILE_NAMES,
    FAILURE_POLICIES,
    CheckpointRequest,
    FitConfig,
    RunConfig,
    resolve_fit_config,
    resolve_run_config_mapping,
)


CONFIG_SCHEMA_VERSION = 1


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--outdir", default="clipp2_results")
    parser.add_argument("--profile", choices=COMPUTATION_PROFILE_NAMES)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--failure-policy", choices=FAILURE_POLICIES)
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="Save an exact-resume checkpoint after each completed transaction.",
    )
    parser.add_argument(
        "--resume",
        metavar="PATH",
        help="Resume and continue updating an identity-matched checkpoint.",
    )
    parser.add_argument(
        "--config",
        metavar="JSON",
        help="Versioned JSON file containing expert fit and runner settings.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipp2",
        description=(
            "Fit certified pairwise-fusion candidates and select an immutable "
            "Ward/CEM partition by fixed-partition BIC."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser(
        "fit", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_fit_args(fit_parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fit" and args.checkpoint and args.resume:
        parser.error("--checkpoint and --resume are mutually exclusive")
    return args


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Configuration repeats JSON key: {key}")
        value[key] = item
    return value


def _load_config(path: str | None) -> tuple[dict[str, object], dict[str, object]]:
    if path is None:
        return {}, {}
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Configuration must be a file: {source}")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Invalid JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read CliPP2 JSON configuration: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("CliPP2 configuration must be a JSON object.")
    allowed = {"schema_version", "fit", "run"}
    if set(payload) - allowed:
        raise ValueError(
            "Unknown configuration section(s): "
            + ", ".join(sorted(set(payload) - allowed))
        )
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported configuration schema_version; required value is "
            f"{CONFIG_SCHEMA_VERSION}."
        )
    fit = payload.get("fit", {})
    run = payload.get("run", {})
    if not isinstance(fit, dict) or not isinstance(run, dict):
        raise ValueError("Configuration fit and run sections must be JSON objects.")
    if "graph" in fit:
        raise ValueError("The graph override is available only to the Python API.")
    return dict(fit), dict(run)


def _resolved_cli(args: argparse.Namespace) -> tuple[FitConfig, RunConfig]:
    fit_values, run_values = _load_config(args.config)
    if args.profile is not None:
        fit_values["computation_profile"] = args.profile
    if args.device is not None:
        fit_values["device"] = args.device
    if args.failure_policy is not None:
        run_values["failure_policy"] = args.failure_policy
    run_config = resolve_run_config_mapping(run_values)
    return resolve_fit_config(**fit_values), run_config


def _fit_config_from_args(args: argparse.Namespace) -> FitConfig:
    """Resolve the single FitConfig used by programmatic CLI tests."""

    return _resolved_cli(args)[0]


def _printable_summary(value: object) -> object:
    """Replace non-finite floats before emitting a Python-literal summary."""

    if isinstance(value, dict):
        return {key: _printable_summary(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_printable_summary(entry) for entry in value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def process_tumor(
    *,
    tumor_file: Path,
    outdir: Path,
    fit_config: FitConfig,
    run_config: RunConfig,
    checkpoint: CheckpointRequest,
) -> dict[str, object]:
    """Lazily enter the heavy inference stack."""

    from .runners.pipeline import process_tumor as run

    return run(
        tumor_file=tumor_file,
        outdir=outdir,
        fit_config=fit_config,
        run_config=run_config,
        checkpoint=checkpoint,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fit_config, run_config = _resolved_cli(args)
    summary = process_tumor(
        tumor_file=Path(args.input_file),
        outdir=Path(args.outdir),
        fit_config=fit_config,
        run_config=run_config,
        checkpoint=CheckpointRequest(
            path=args.resume or args.checkpoint,
            enabled=bool(args.checkpoint or args.resume),
            resume=args.resume is not None,
        ),
    )
    print(_printable_summary(summary))


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
