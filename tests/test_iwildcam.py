from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from datasets import Dataset, Image  # type: ignore
    from PIL import Image as PILImage  # type: ignore
except ImportError:  # pragma: no cover - optional data dependencies
    Dataset = None
    Image = None
    PILImage = None

from dabench.data.iwildcam import enrich_iwildcam_split, get_iwildcam_class_names


@unittest.skipIf(Dataset is None or Image is None or PILImage is None, "datasets and Pillow are required")
class IWildCamMetadataTest(unittest.TestCase):
    def test_enrich_train_split_with_sparse_category_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.json"
            test_path = root / "test.json"
            train_path.write_text(
                json.dumps(
                    {
                        "categories": [
                            {"id": 0, "name": "empty"},
                            {"id": 73, "name": "tapirus bairdii"},
                        ],
                        "images": [
                            {
                                "id": "img-1",
                                "file_name": "img-1.jpg",
                                "location": 5,
                                "seq_id": "seq-1",
                                "frame_num": 2,
                                "seq_num_frames": 3,
                                "datetime": "2020-01-01 00:00:00.000",
                                "width": 8,
                                "height": 8,
                            }
                        ],
                        "annotations": [
                            {"image_id": "img-1", "category_id": 73, "id": "ann-1", "count": 1}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            test_path.write_text(json.dumps({"categories": [], "images": []}), encoding="utf-8")
            paths_path = root / "paths.json"
            paths_path.write_text(
                json.dumps(
                    {
                        "datasets": {
                            "iwildcam": {
                                "train_annotations_path": str(train_path),
                                "test_information_path": str(test_path),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            dataset = Dataset.from_dict({"image": [{"bytes": _jpeg_bytes(), "path": "img-1.jpg"}]})
            dataset = dataset.cast_column("image", Image(decode=False))

            with patch.dict(os.environ, {"DABENCH_PATHS_FILE": str(paths_path)}, clear=False):
                enriched = enrich_iwildcam_split(dataset, split="train", decode=False)
                self.assertEqual(enriched[0]["label"], 1)
                self.assertEqual(enriched[0]["category_id"], 73)
                self.assertEqual(enriched[0]["category_name"], "tapirus bairdii")
                self.assertEqual(enriched[0]["location"], 5)
                self.assertEqual(enriched.features["label"].dtype, "int64")
                self.assertEqual(get_iwildcam_class_names(), ("empty", "tapirus bairdii"))


def _jpeg_bytes() -> bytes:
    image = PILImage.new("RGB", (8, 8), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
