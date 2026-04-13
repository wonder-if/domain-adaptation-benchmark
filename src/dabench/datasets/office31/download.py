"""Download helpers for Office-31 from the ModelScope Git LFS repository."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from dabench.datasets.hf_download import proxy_env
from dabench.datasets.office31.load import inspect_dataset as inspect_image_dataset
from dabench.io import ensure_dir

REPO_URL = "https://www.modelscope.cn/datasets/OmniData/Office-31.git"
IMAGE_ARCHIVE = "domain_adaptation_images.tar.gz"


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _check_tools() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("Office-31 download requires `git`.")
    _run(["git", "lfs", "version"])


def _safe_extract_tar(archive: Path, dest: Path, *, force: bool) -> dict[str, Any]:
    if dest.exists() and any(dest.iterdir()) and not force:
        return {"archive": str(archive), "dest": str(dest), "members": None, "skipped": True}
    if force and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            member_path = (dest / member.name).resolve()
            if not str(member_path).startswith(str(dest.resolve()) + os.sep):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        handle.extractall(dest)
    return {"archive": str(archive), "dest": str(dest), "members": len(members), "skipped": False}


def _link_images(extract_dir: Path, dest: Path, *, force: bool, symlink: bool) -> dict[str, Any]:
    if force and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    domains = ["amazon", "dslr", "webcam"]
    created: list[str] = []
    skipped: list[str] = []
    for domain in domains:
        source = extract_dir / domain / "images"
        target = dest / domain
        if target.exists():
            skipped.append(domain)
            continue
        if not source.is_dir():
            raise FileNotFoundError(f"Missing extracted Office-31 domain directory: {source}")
        if symlink:
            target.symlink_to(source, target_is_directory=True)
        else:
            shutil.copytree(source, target)
        created.append(domain)

    return {
        "dest": str(dest),
        "domains": domains,
        "created": created,
        "skipped": skipped,
        "symlink": symlink,
    }


def download_dataset(
    *,
    dest: str | Path,
    proxy: str = "disable",
    repo_url: str = REPO_URL,
    clone_dir: str | Path | None = None,
    extract_dir: str | Path | None = None,
    revision: str | None = None,
    force: bool = False,
    symlink: bool = True,
) -> dict[str, Any]:
    """Download Office-31 via ModelScope Git LFS and prepare a folder dataset.

    `dest` is the final training directory with `amazon/`, `dslr/`, and `webcam/`
    subdirectories. By default these are symlinks into the extracted archive, so
    the image files are not duplicated.
    """

    destination = Path(dest).expanduser().resolve()
    default_parent = destination.parent
    resolved_clone_dir = (
        Path(clone_dir).expanduser().resolve() if clone_dir else default_parent / f"{destination.name}_git"
    )
    resolved_extract_dir = (
        Path(extract_dir).expanduser().resolve() if extract_dir else default_parent / f"{destination.name}_extracted"
    )

    with proxy_env(proxy):
        _check_tools()
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"

        if force and resolved_clone_dir.exists():
            shutil.rmtree(resolved_clone_dir)
        if not resolved_clone_dir.exists():
            ensure_dir(resolved_clone_dir.parent)
            _run(["git", "clone", repo_url, str(resolved_clone_dir)], env=env)
        if revision:
            _run(["git", "-C", str(resolved_clone_dir), "checkout", revision])
        _run(["git", "-C", str(resolved_clone_dir), "lfs", "pull"])

    archive = resolved_clone_dir / "raw" / IMAGE_ARCHIVE
    if not archive.is_file():
        raise FileNotFoundError(f"Missing Office-31 image archive after Git LFS pull: {archive}")

    extract_info = _safe_extract_tar(archive, resolved_extract_dir, force=force)
    link_info = _link_images(resolved_extract_dir, destination, force=force, symlink=symlink)
    dataset_info = inspect_image_dataset(path=destination)

    return {
        "dataset": "office-31",
        "source": "modelscope-git-lfs",
        "repo_url": repo_url,
        "clone_dir": str(resolved_clone_dir),
        "extract_dir": str(resolved_extract_dir),
        "dest": str(destination),
        "revision": revision,
        "archive": str(archive),
        "extract": extract_info,
        "organize": link_info,
        "prepared": dataset_info,
    }


def inspect_dataset(*, path: str | Path) -> dict[str, Any]:
    return inspect_image_dataset(path=path)
