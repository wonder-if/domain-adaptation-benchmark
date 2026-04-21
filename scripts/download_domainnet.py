#!/usr/bin/env python3
"""Minimal DomainNet download example."""

from __future__ import annotations

import argparse
import json

from dabench.storage import download_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download DomainNet prepared data.")
    parser.add_argument("--dest", required=True, help="Prepared dataset directory to create.")
    parser.add_argument("--source", choices=["mirror", "hf"], default="mirror")
    parser.add_argument("--config", default=None, help="Optional DomainNet config name.")
    parser.add_argument("--cache-dir", default=None, help="Optional datasets cache directory.")
    parser.add_argument("--revision", default=None, help="Optional dataset revision.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--file-format", choices=["arrow", "parquet"], default="arrow")
    parser.add_argument(
        "--verification-mode",
        choices=["all_checks", "basic_checks", "no_checks"],
        default=None,
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force datasets to redownload and rebuild the prepared directory.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = download_dataset(
        "domainnet",
        dest=args.dest,
        source=args.source,
        config=args.config,
        cache_dir=args.cache_dir,
        revision=args.revision,
        token=args.token,
        num_proc=args.num_proc,
        file_format=args.file_format,
        verification_mode=args.verification_mode,
        force_redownload=args.force_redownload,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
