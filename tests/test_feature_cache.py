from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
except ImportError:  # pragma: no cover - optional data dependencies
    torch = None
    Dataset = None

from dabench.data import (
    default_feature_collator,
    list_feature_caches,
    load_feature_decode_errors,
    load_feature_view,
    load_text_features,
)


@unittest.skipIf(torch is None or Dataset is None, "datasets and torch are required")
class FeatureCacheTest(unittest.TestCase):
    def test_load_feature_view_and_text_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_root = root / "office-31" / "amazon" / "image_features" / "clip-vit-base-patch16"
            text_root = root / "office-31" / "amazon" / "text_features"
            image_root.parent.mkdir(parents=True)
            text_root.mkdir(parents=True)

            dataset = Dataset.from_dict(
                {
                    "label": [0, 1],
                    "image_features": [[1.0, 0.0], [0.0, 1.0]],
                }
            )
            dataset.save_to_disk(str(image_root))
            torch.save({"text_features": torch.ones(2, 2)}, text_root / "clip-vit-base-patch16-a photo.pt")

            loaded = load_feature_view(
                "office-31",
                domain="amazon",
                feature_model="clip-vit-base-patch16",
                feature_cache_path=root,
            )
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["domain"], "amazon")
            self.assertEqual(loaded[0]["image_features"], [1.0, 0.0])

            text_features = load_text_features(
                "office-31",
                domain="amazon",
                feature_model="clip-vit-base-patch16",
                prompt="a photo",
                feature_cache_path=root,
            )
            self.assertEqual(tuple(text_features["text_features"].shape), (2, 2))

            split_text_features = load_text_features(
                "office-31",
                split="amazon",
                feature_model="clip-vit-base-patch16",
                prompt="a photo",
                feature_cache_path=root,
            )
            self.assertEqual(tuple(split_text_features["text_features"].shape), (2, 2))

            batch = default_feature_collator([loaded[0], loaded[1]])
            self.assertEqual(tuple(batch["image_features"].shape), (2, 2))
            self.assertEqual(batch["labels"].tolist(), [0, 1])

    def test_list_split_feature_cache_stats_and_decode_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_root = root / "camelyon17" / "id_val" / "image_features" / "clip-vit-base-patch16"
            text_root = root / "camelyon17" / "id_val" / "text_features"
            image_root.parent.mkdir(parents=True)
            text_root.mkdir(parents=True)

            Dataset.from_dict(
                {
                    "label": [0, 1, 0],
                    "image_features": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                }
            ).save_to_disk(str(image_root))
            (image_root / "decode_errors.json").write_text(
                json.dumps([{"row_index": 9, "image_id": "bad"}]),
                encoding="utf-8",
            )
            torch.save({"text_features": torch.ones(2, 2)}, text_root / "clip-vit-base-patch16-a photo.pt")

            records = list_feature_caches("camelyon17", feature_cache_path=root, include_stats=True)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["axis"], "split")
            self.assertEqual(records[0]["split"], "id_val")
            self.assertEqual(records[0]["domain"], "id_val")
            self.assertEqual(records[0]["image_model_stats"]["clip-vit-base-patch16"]["rows"], 3)
            self.assertEqual(records[0]["image_model_stats"]["clip-vit-base-patch16"]["feature_dim"], 2)
            self.assertEqual(records[0]["image_model_stats"]["clip-vit-base-patch16"]["decode_errors"], 1)

            errors = load_feature_decode_errors(
                "camelyon17",
                split="id_val",
                feature_model="clip-vit-base-patch16",
                feature_cache_path=root,
            )
            self.assertEqual(errors[0]["row_index"], 9)

    def test_minidomainnet_feature_view_uses_independent_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_root = root / "minidomainnet" / "clipart" / "image_features" / "clip-vit-base-patch16"
            text_root = root / "minidomainnet" / "clipart" / "text_features"
            split_dir = root / "splits"
            prepared_root = root / "prepared-domainnet"
            image_root.parent.mkdir(parents=True)
            text_root.mkdir(parents=True)
            split_dir.mkdir()
            prepared_root.mkdir()

            Dataset.from_dict(
                {
                    "label": [0, 1, 0],
                    "image_path": [
                        "clipart/class_a/img1.jpg",
                        "clipart/class_b/img2.jpg",
                        "clipart/class_a/img3.jpg",
                    ],
                    "image_features": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                }
            ).save_to_disk(str(image_root))
            (split_dir / "clipart_train.txt").write_text(
                "clipart/class_a/img3.jpg 0\nclipart/class_a/img1.jpg 0\n",
                encoding="utf-8",
            )
            torch.save({"text_features": torch.ones(2, 2)}, text_root / "clip-vit-base-patch16-a photo.pt")

            paths_file = root / "paths.json"
            paths_file.write_text(
                json.dumps(
                    {
                        "datasets": {
                            "minidomainnet": {
                                "path": str(prepared_root),
                                "split_dir": str(split_dir),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            old_paths_file = os.environ.get("DABENCH_PATHS_FILE")
            os.environ["DABENCH_PATHS_FILE"] = str(paths_file)
            try:
                loaded = load_feature_view(
                    "minidomainnet",
                    domain="clipart",
                    split="train",
                    feature_model="clip-vit-base-patch16",
                    feature_cache_path=root,
                )
                self.assertEqual(len(loaded), 2)
                self.assertEqual(loaded[0]["domain"], "clipart")
                self.assertEqual(loaded["image_path"], ["clipart/class_a/img1.jpg", "clipart/class_a/img3.jpg"])
                self.assertEqual(loaded["label"], [0, 0])
                self.assertEqual(loaded["image_features"], [[1.0, 0.0], [0.5, 0.5]])

                text_features = load_text_features(
                    "minidomainnet",
                    domain="clipart",
                    feature_model="clip-vit-base-patch16",
                    prompt="a photo",
                    feature_cache_path=root,
                )
                self.assertEqual(tuple(text_features["text_features"].shape), (2, 2))

                records = list_feature_caches("minidomainnet", feature_cache_path=root)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["dataset"], "minidomainnet")
                self.assertNotIn("cache_dataset", records[0])
                self.assertEqual(records[0]["domain"], "clipart")
            finally:
                if old_paths_file is None:
                    os.environ.pop("DABENCH_PATHS_FILE", None)
                else:
                    os.environ["DABENCH_PATHS_FILE"] = old_paths_file


if __name__ == "__main__":
    unittest.main()
