"""CLI entrypoint for dabench."""

from __future__ import annotations

import argparse
import json

from dabench.suite import get_suite, list_suites
from dabench.storage import download_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dabench", description="Domain adaptation benchmark utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download dataset artifacts.")
    download_subparsers = download_parser.add_subparsers(dest="dataset", required=True)
    domainnet_download = download_subparsers.add_parser("domainnet", help="Download DomainNet via datasets.")
    domainnet_download.add_argument("--dest", required=True, help="Target prepared dataset directory.")
    domainnet_download.add_argument("--source", choices=["mirror", "hf"], default="mirror")
    domainnet_download.add_argument("--config", default=None, help="Optional DomainNet config name.")
    domainnet_download.add_argument("--proxy", choices=["keep", "disable"], default="disable")
    domainnet_download.add_argument("--cache-dir", default=None, help="Optional datasets cache directory.")
    domainnet_download.add_argument("--revision", default=None, help="Optional dataset revision.")
    domainnet_download.add_argument("--token", default=None, help="Optional Hugging Face token.")
    domainnet_download.add_argument("--num-proc", type=int, default=None)
    domainnet_download.add_argument("--file-format", choices=["arrow", "parquet"], default="arrow")
    domainnet_download.add_argument(
        "--verification-mode",
        choices=["all_checks", "basic_checks", "no_checks"],
        default=None,
    )
    domainnet_download.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force datasets to redownload and rebuild the prepared directory.",
    )

    iwildcam_download = download_subparsers.add_parser("iwildcam", help="Download iWildCam via datasets.")
    iwildcam_download.add_argument("--dest", required=True, help="Target prepared dataset directory.")
    iwildcam_download.add_argument("--source", choices=["mirror", "hf"], default="mirror")
    iwildcam_download.add_argument("--proxy", choices=["keep", "disable"], default="disable")
    iwildcam_download.add_argument("--cache-dir", default=None, help="Optional datasets cache directory.")
    iwildcam_download.add_argument("--revision", default=None, help="Optional dataset revision.")
    iwildcam_download.add_argument("--token", default=None, help="Optional Hugging Face token.")
    iwildcam_download.add_argument("--num-proc", type=int, default=None)
    iwildcam_download.add_argument("--file-format", choices=["arrow", "parquet"], default="arrow")
    iwildcam_download.add_argument(
        "--verification-mode",
        choices=["all_checks", "basic_checks", "no_checks"],
        default=None,
    )
    iwildcam_download.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force datasets to redownload and rebuild the prepared directory.",
    )

    office31_download = download_subparsers.add_parser("office-31", help="Download Office-31 via ModelScope Git LFS.")
    office31_download.add_argument("--dest", required=True, help="Final prepared directory with amazon/dslr/webcam.")
    office31_download.add_argument("--proxy", choices=["keep", "disable"], default="disable")
    office31_download.add_argument("--clone-dir", default=None, help="Optional Git LFS clone directory.")
    office31_download.add_argument("--extract-dir", default=None, help="Optional archive extraction directory.")
    office31_download.add_argument("--revision", default=None, help="Optional Git revision to checkout.")
    office31_download.add_argument("--force", action="store_true", help="Recreate clone/extract/prepared directories.")
    office31_download.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking domains.")

    tasks_parser = subparsers.add_parser("tasks", help="List and inspect benchmark task suites.")
    tasks_subparsers = tasks_parser.add_subparsers(dest="tasks_command", required=True)
    tasks_subparsers.add_parser("list", help="List built-in benchmark suites.")
    tasks_show = tasks_subparsers.add_parser("show", help="Show tasks in a built-in benchmark suite.")
    tasks_show.add_argument("suite_id", help="Built-in suite id, for example office31_closed_set_uda.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "download" and args.dataset == "domainnet":
        result = download_dataset(
            "domainnet",
            dest=args.dest,
            source=args.source,
            config=args.config,
            proxy=args.proxy,
            cache_dir=args.cache_dir,
            revision=args.revision,
            token=args.token,
            num_proc=args.num_proc,
            file_format=args.file_format,
            verification_mode=args.verification_mode,
            force_redownload=args.force_redownload,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "download" and args.dataset == "iwildcam":
        download_dataset(
            "iwildcam",
            dest=args.dest,
            source=args.source,
            proxy=args.proxy,
            cache_dir=args.cache_dir,
            revision=args.revision,
            token=args.token,
            num_proc=args.num_proc,
            file_format=args.file_format,
            verification_mode=args.verification_mode,
            force_redownload=args.force_redownload,
        )
        return

    if args.command == "download" and args.dataset == "office-31":
        result = download_dataset(
            "office-31",
            dest=args.dest,
            proxy=args.proxy,
            clone_dir=args.clone_dir,
            extract_dir=args.extract_dir,
            revision=args.revision,
            force=args.force,
            symlink=not args.copy_images,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "tasks" and args.tasks_command == "list":
        payload = [
            {
                "suite_id": suite.suite_id,
                "name": suite.name,
                "num_tasks": len(suite.tasks),
                "metadata": dict(suite.metadata),
            }
            for suite in list_suites()
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "tasks" and args.tasks_command == "show":
        suite = get_suite(args.suite_id)
        print(json.dumps(suite.to_dict(), indent=2, sort_keys=True))
        return

    parser.error("unsupported command")


if __name__ == "__main__":
    main()
