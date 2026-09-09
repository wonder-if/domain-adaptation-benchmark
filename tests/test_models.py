from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dabench.models import (
    MODEL_ROOT_ENV,
    get_model_root,
    infer_loader,
    list_model_names,
    resolve_model_path,
)
from dabench.models.registry import get_model_spec


class ModelRegistryTest(unittest.TestCase):
    def test_registry_exposes_local_aliases(self) -> None:
        names = list_model_names()
        self.assertIn("clip-vit-base-patch16", names)
        self.assertIn("clip-vit-l-14-datacomp.xl-s13b-b90k", names)
        self.assertIn("resnet-50", names)
        self.assertEqual(get_model_spec("clip-vit-l-14-datacomp.xl-s13b-b90k").loader, "transformers")

    def test_model_root_honors_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {MODEL_ROOT_ENV: tmpdir}):
                self.assertEqual(get_model_root(), Path(tmpdir).resolve())

    def test_model_root_uses_path_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths_file = root / "paths.json"
            model_root = root / "models"
            paths_file.write_text(json.dumps({"models": {"root": str(model_root)}}), encoding="utf-8")

            with patch.dict(os.environ, {"DABENCH_PATHS_FILE": str(paths_file)}, clear=False):
                self.assertEqual(get_model_root(), model_root.resolve())

    def test_resolve_registered_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = get_model_spec("clip-vit-base-patch16")
            expected = root / spec.relative_path
            expected.mkdir(parents=True)

            self.assertEqual(resolve_model_path(spec.name, model_root=root), expected)

    def test_resolve_unknown_model_reports_available_aliases(self) -> None:
        with self.assertRaisesRegex(ValueError, "Available models"):
            resolve_model_path("missing-model")

    def test_infer_loader_from_local_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            datacomp_dir = root / "datacomp"
            datacomp_dir.mkdir()
            (datacomp_dir / "open_clip_config.json").write_text("{}", encoding="utf-8")
            (datacomp_dir / "config.json").write_text(json.dumps({"model_type": "clip"}), encoding="utf-8")
            self.assertEqual(infer_loader(datacomp_dir), "transformers")

            timm_dir = root / "timm"
            timm_dir.mkdir()
            (timm_dir / "config.json").write_text(
                json.dumps({"architecture": "eva02_base_patch14_224", "pretrained_cfg": {}}),
                encoding="utf-8",
            )
            self.assertEqual(infer_loader(timm_dir), "timm")

            transformers_dir = root / "transformers"
            transformers_dir.mkdir()
            (transformers_dir / "config.json").write_text(
                json.dumps({"model_type": "clip"}),
                encoding="utf-8",
            )
            self.assertEqual(infer_loader(transformers_dir), "transformers")


if __name__ == "__main__":
    unittest.main()
