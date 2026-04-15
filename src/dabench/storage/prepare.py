"""Unified storage preparation entrypoints."""

from __future__ import annotations

import os
import shutil
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dabench.storage.manifest import get_manifest
from dabench.storage.paths import set_dataset_path
from dabench.utils.commands import run_command
from dabench.utils.imports import require_requests
from dabench.utils.paths import ensure_dir, expand_path

PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy")
HF_ENDPOINTS = {"hf": "https://huggingface.co", "mirror": "https://hf-mirror.com"}


@contextmanager
def _proxy_env(mode: str) -> Iterator[None]:
    if mode not in {"keep", "disable"}:
        raise ValueError("proxy must be one of: keep, disable")
    if mode == "keep":
        yield
        return

    original = {key: os.environ.get(key) for key in PROXY_KEYS}
    for key in PROXY_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _hf_endpoint(source: str) -> Iterator[None]:
    if source not in HF_ENDPOINTS:
        raise ValueError("source must be one of: hf, mirror")
    original = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = HF_ENDPOINTS[source]
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = original


def _require_hf_datasets():
    try:
        from datasets import DownloadConfig, DownloadMode, load_dataset_builder  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for Hugging Face dataset storage preparation. Install with `pip install -e .[data]`."
        ) from exc
    return load_dataset_builder, DownloadConfig, DownloadMode


def _prepare_hf(
    manifest: dict[str, Any],
    *,
    dest: str | Path,
    source: str | None = None,
    config: str | None = None,
    proxy: str = "disable",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    num_proc: int | None = None,
    file_format: str = "arrow",
    verification_mode: str | None = None,
    force_redownload: bool = False,
) -> dict[str, Any]:
    storage = manifest["storage"]
    dataset_name = storage.get("dataset_name")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(f"Manifest {manifest['id']!r} must define storage.dataset_name for hf backend.")

    destination = ensure_dir(dest)
    resolved_cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
    resolved_source = source or str(storage.get("default_source", "mirror"))
    kwargs: dict[str, Any] = {}
    if config:
        kwargs["name"] = config
    if resolved_cache_dir:
        kwargs["cache_dir"] = str(resolved_cache_dir)
    if revision:
        kwargs["revision"] = revision
    if token is not None:
        kwargs["token"] = token

    with _proxy_env(proxy), _hf_endpoint(resolved_source):
        load_dataset_builder, DownloadConfig, DownloadMode = _require_hf_datasets()
        builder = load_dataset_builder(dataset_name, download_config=DownloadConfig(), **kwargs)
        builder.download_and_prepare(
            output_dir=str(destination),
            file_format=file_format,
            num_proc=num_proc,
            verification_mode=verification_mode,
            download_mode=DownloadMode.FORCE_REDOWNLOAD if force_redownload else None,
        )

    return {
        "dataset": manifest["id"],
        "backend": "hf",
        "source_dataset": dataset_name,
        "source": resolved_source,
        "config": builder.config.name,
        "dest": str(destination),
        "cache_dir": str(resolved_cache_dir) if resolved_cache_dir else None,
        "splits": sorted(builder.info.splits.keys()) if builder.info.splits else [],
        "file_format": file_format,
    }


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


def _prepare_ms_office31(
    manifest: dict[str, Any],
    *,
    dest: str | Path,
    proxy: str = "disable",
    clone_dir: str | Path | None = None,
    extract_dir: str | Path | None = None,
    repo_url: str | None = None,
    revision: str | None = None,
    force: bool = False,
    symlink: bool = True,
) -> dict[str, Any]:
    storage = manifest["storage"]
    manifest_repo_url = storage.get("repo_url")
    image_archive = storage.get("image_archive")
    if not isinstance(manifest_repo_url, str) or not manifest_repo_url:
        raise ValueError(f"Manifest {manifest['id']!r} must define storage.repo_url.")
    if repo_url is not None and repo_url != manifest_repo_url:
        raise ValueError("Custom Office-31 repo_url is no longer supported here; update manifests/office-31.yaml instead.")
    if not isinstance(image_archive, str) or not image_archive:
        raise ValueError(f"Manifest {manifest['id']!r} must define storage.image_archive.")
    if shutil.which("git") is None:
        raise RuntimeError("ModelScope storage preparation requires `git`.")
    if shutil.which("git-lfs") is None:
        raise RuntimeError("ModelScope storage preparation requires `git-lfs` on PATH.")

    destination = expand_path(dest)
    resolved_clone_dir = expand_path(clone_dir) if clone_dir else destination.parent / f"{destination.name}_git"
    resolved_extract_dir = expand_path(extract_dir) if extract_dir else destination.parent / f"{destination.name}_extracted"

    with _proxy_env(proxy):
        run_command(["git", "lfs", "version"])
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        if force and resolved_clone_dir.exists():
            shutil.rmtree(resolved_clone_dir)
        if not resolved_clone_dir.exists():
            ensure_dir(resolved_clone_dir.parent)
            run_command(["git", "clone", manifest_repo_url, str(resolved_clone_dir)], env=env)
        if revision:
            run_command(["git", "-C", str(resolved_clone_dir), "checkout", revision])
        run_command(["git", "-C", str(resolved_clone_dir), "lfs", "pull"])

    archive = resolved_clone_dir / "raw" / image_archive
    if not archive.is_file():
        raise FileNotFoundError(f"Missing Office-31 image archive after Git LFS pull: {archive}")

    extract_info = _safe_extract_tar(archive, resolved_extract_dir, force=force)
    destination.mkdir(parents=True, exist_ok=True)
    domains = manifest.get("prepared", {}).get("domains", [])
    created: list[str] = []
    skipped: list[str] = []
    for domain in domains:
        if not isinstance(domain, str):
            raise ValueError(f"Manifest {manifest['id']!r} prepared.domains must contain only strings.")
        source_dir = resolved_extract_dir / domain / "images"
        target_dir = destination / domain
        if target_dir.exists():
            skipped.append(domain)
            continue
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing extracted Office-31 domain directory: {source_dir}")
        if symlink:
            target_dir.symlink_to(source_dir, target_is_directory=True)
        else:
            shutil.copytree(source_dir, target_dir)
        created.append(domain)

    return {
        "dataset": manifest["id"],
        "backend": "ms",
        "repo_url": manifest_repo_url,
        "clone_dir": str(resolved_clone_dir),
        "extract_dir": str(resolved_extract_dir),
        "dest": str(destination),
        "revision": revision,
        "archive": str(archive),
        "extract": extract_info,
        "organize": {"domains": domains, "created": created, "skipped": skipped, "symlink": symlink},
    }


def _download_file(url: str, dest: Path, *, proxy: str, force: bool) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return {"url": url, "path": str(dest), "size": dest.stat().st_size, "skipped": True}
    requests = require_requests()
    with _proxy_env(proxy), requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return {"url": url, "path": str(dest), "size": dest.stat().st_size, "skipped": False}


def _extract_tar(archive: Path, dest: Path, *, force: bool) -> dict[str, Any]:
    output_dir = ensure_dir(dest)
    with tarfile.open(archive, "r") as handle:
        members = handle.getmembers()
        top_level = {Path(member.name).parts[0] for member in members if member.name and Path(member.name).parts}
        if top_level and not force:
            targets = [output_dir / name for name in top_level]
            if all(target.exists() and (not target.is_dir() or any(target.iterdir())) for target in targets):
                return {"archive": str(archive), "dest": str(output_dir), "members": len(members), "skipped": True}
        handle.extractall(output_dir)
    return {"archive": str(archive), "dest": str(output_dir), "members": len(members), "skipped": False}


def _prepare_visda2017(
    manifest: dict[str, Any],
    *,
    dest: str | Path,
    proxy: str = "disable",
    force: bool = False,
    extract: bool = True,
) -> dict[str, Any]:
    storage = manifest["storage"]
    archives = storage.get("archives")
    if not isinstance(archives, dict):
        raise ValueError(f"Manifest {manifest['id']!r} must define storage.archives for handler visda2017.")

    destination = ensure_dir(dest)
    archives_dir = ensure_dir(destination / "archives")
    extracted_dir = ensure_dir(destination / "data")
    downloaded: dict[str, object] = {}
    extracted: dict[str, object] = {}
    for split_name, url in archives.items():
        if not isinstance(split_name, str) or not isinstance(url, str):
            raise ValueError(f"Manifest {manifest['id']!r} storage.archives must map split names to URLs.")
        archive_path = archives_dir / f"{split_name}.tar"
        downloaded[split_name] = _download_file(url, archive_path, proxy=proxy, force=force)
        if extract:
            extracted[split_name] = _extract_tar(archive_path, extracted_dir, force=force)

    image_list = None
    image_list_url = storage.get("image_list_url")
    if isinstance(image_list_url, str) and image_list_url:
        image_list = _download_file(image_list_url, destination / "image_list.txt", proxy=proxy, force=force)

    return {
        "dataset": manifest["id"],
        "backend": "other",
        "handler": "visda2017",
        "dest": str(destination),
        "archives": downloaded,
        "extracted": extracted,
        "image_list": image_list,
    }


def prepare_dataset(name: str, *, dest: str | Path, **kwargs: Any) -> dict[str, Any]:
    manifest = get_manifest(name)
    backend = manifest["storage"]["backend"]
    if backend == "hf":
        result = _prepare_hf(manifest, dest=dest, **kwargs)
    elif backend == "ms":
        result = _prepare_ms_office31(manifest, dest=dest, **kwargs)
    elif backend == "other" and manifest["storage"].get("handler") == "visda2017":
        result = _prepare_visda2017(manifest, dest=dest, **kwargs)
    else:
        raise ValueError(f"Unsupported storage backend for {manifest['id']!r}: {backend!r}")
    config_path = set_dataset_path(manifest["id"], result["dest"])
    result["config_path"] = str(config_path)
    return result
