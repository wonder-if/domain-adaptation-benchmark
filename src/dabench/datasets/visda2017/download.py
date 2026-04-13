"""Download helpers for the official VisDA-2017 classification dataset."""

from __future__ import annotations

from pathlib import Path

from dabench.datasets.url_download import download_file, extract_tar
from dabench.io import ensure_dir

ARCHIVES = {
    "train": "http://csr.bu.edu/ftp/visda17/clf/train.tar",
    "validation": "http://csr.bu.edu/ftp/visda17/clf/validation.tar",
    "test": "http://csr.bu.edu/ftp/visda17/clf/test.tar",
}

IMAGE_LIST_URL = (
    "https://raw.githubusercontent.com/"
    "VisionLearningGroup/taskcv-2017-public/master/classification/data/image_list.txt"
)


def download_dataset(
    *,
    dest: str | Path,
    proxy: str = "disable",
    force: bool = False,
    extract: bool = True,
) -> dict[str, object]:
    destination = ensure_dir(dest)
    archives_dir = ensure_dir(destination / "archives")
    extracted_dir = ensure_dir(destination / "data")

    archives: dict[str, object] = {}
    extracted: dict[str, object] = {}
    for split_name, url in ARCHIVES.items():
        archive_path = archives_dir / f"{split_name}.tar"
        archives[split_name] = download_file(
            url=url,
            dest=archive_path,
            proxy=proxy,
            force=force,
        )
        if extract:
            extracted[split_name] = extract_tar(
                archive_path=archive_path,
                dest=extracted_dir,
                force=force,
            )

    image_list = download_file(
        url=IMAGE_LIST_URL,
        dest=destination / "image_list.txt",
        proxy=proxy,
        force=force,
    )
    return {
        "dataset": "visda-2017",
        "dest": str(destination),
        "archives": archives,
        "extracted": extracted,
        "image_list": image_list,
    }


def inspect_dataset(*, path: str | Path) -> dict[str, object]:
    target = Path(path).expanduser().resolve()
    data_dir = target / "data"
    return {
        "path": str(target),
        "exists": target.exists(),
        "has_data_dir": data_dir.is_dir(),
        "train_dir": str(data_dir / "train"),
        "validation_dir": str(data_dir / "validation"),
        "test_dir": str(data_dir / "test"),
        "has_image_list": (target / "image_list.txt").is_file(),
    }
