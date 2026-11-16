import importlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def _result(boxes_xyxy):
    boxes = MagicMock()
    boxes.__len__.return_value = len(boxes_xyxy)
    boxes.xyxy.cpu.return_value.numpy.return_value = np.asarray(boxes_xyxy, dtype=float)
    boxes.conf.cpu.return_value.numpy.return_value = np.full(len(boxes_xyxy), 0.8)
    return SimpleNamespace(boxes=boxes)


class DetectionAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep real request parsing, image decoding and selection, but avoid
        # loading model weights or running inference in deterministic API tests.
        fake_yolo = SimpleNamespace(YOLO=Mock(return_value=Mock(names={2: "car"})))
        with patch.dict("sys.modules", {"ultralytics": fake_yolo}):
            cls.service = importlib.import_module("app")
        image = io.BytesIO()
        Image.new("RGB", (2000, 1000)).save(image, format="PNG")
        cls.image_bytes = image.getvalue()

    def setUp(self):
        self.client = TestClient(self.service.app)

    def tearDown(self):
        self.client.close()

    def _post(self, endpoint, predictions, params):
        field = "files" if endpoint == "/detect-batch" else "file"
        with patch.object(self.service.model, "predict", side_effect=predictions) as predict:
            response = self.client.post(
                endpoint, params=params,
                files=[(field, ("parking.png", self.image_bytes, "image/png"))],
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(predict.call_count, len(predictions))
        payload = response.json()
        return payload["items"][0] if endpoint == "/detect-batch" else payload

    def test_both_endpoints_apply_default_and_configured_vertical_priority(self):
        foreground = [0, 650, 600, 850]
        centered = [700, 400, 1200, 600]

        for endpoint in ("/detect", "/detect-batch"):
            for weight, expected in ((None, centered), (0, foreground), (4, centered)):
                with self.subTest(endpoint=endpoint, weight=weight):
                    params = {"roll_verify": "false"}
                    if weight is not None:
                        params["vertical_center_weight"] = weight
                    payload = self._post(endpoint, [[_result([foreground, centered])]], params)
                    self.assertEqual(payload["bbox_xyxy"], expected)
                    self.assertFalse(payload["wraps_x"])
                    self.assertEqual(payload["app_version"], self.service.APP_VERSION)

    def test_both_endpoints_compare_verified_seam_car_with_original_using_requested_weight(self):
        main = [700, 400, 1200, 600]
        edge = [0, 650, 300, 850]
        rolled_bbox = [700, 650, 1300, 850]
        mapped_bbox = [1700, 650, 300, 850]

        for endpoint in ("/detect", "/detect-batch"):
            for weight, expected in ((1, main), (0, mapped_bbox)):
                with self.subTest(endpoint=endpoint, weight=weight):
                    payload = self._post(
                        endpoint, [[_result([main, edge])], [_result([rolled_bbox])]],
                        {"vertical_center_weight": weight},
                    )
                    self.assertEqual(payload["bbox_xyxy"], expected)
                    self.assertEqual(payload["wraps_x"], weight == 0)

    def test_both_endpoints_keep_large_main_car_when_small_seam_car_is_verified(self):
        main = [600, 200, 1400, 800]
        edge = [0, 500, 120, 800]
        rolled_bbox = [900, 500, 1120, 800]

        for endpoint in ("/detect", "/detect-batch"):
            with self.subTest(endpoint=endpoint):
                payload = self._post(endpoint, [[_result([main, edge])], [_result([rolled_bbox])]], {})
                self.assertEqual(payload["bbox_xyxy"], main)
                self.assertFalse(payload["wraps_x"])

    def test_batch_applies_weight_to_every_image_across_chunks(self):
        foreground = [0, 650, 600, 850]
        centered = [700, 400, 1200, 600]

        for weight, expected in ((1, centered), (0, foreground)):
            with self.subTest(weight=weight):
                results = [_result([foreground, centered]) for _ in range(3)]
                with patch.object(self.service.model, "predict", side_effect=[results[:2], results[2:]]) as predict:
                    response = self.client.post(
                        "/detect-batch",
                        params={"vertical_center_weight": weight, "roll_verify": "false", "chunk_size": 2},
                        files=[("files", (f"parking-{i}.png", self.image_bytes, "image/png")) for i in range(3)],
                    )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(predict.call_count, 2)
                self.assertEqual(response.json()["count"], 3)
                self.assertEqual([item["bbox_xyxy"] for item in response.json()["items"]], [expected] * 3)

    def test_no_detections_preserves_empty_response_and_skips_roll(self):
        for endpoint in ("/detect", "/detect-batch"):
            for result in (_result([]), SimpleNamespace(boxes=None)):
                with self.subTest(endpoint=endpoint, result=result):
                    payload = self._post(endpoint, [[result]], {})
                    self.assertIsNone(payload["bbox_xyxy"])
                    self.assertIsNone(payload["confidence"])
                    self.assertFalse(payload["wraps_x"])

    def test_both_endpoints_reject_invalid_weights_before_inference(self):
        for endpoint in ("/detect", "/detect-batch"):
            for weight in (-0.1, 4.1, "nan", "inf", "invalid"):
                with self.subTest(endpoint=endpoint, weight=weight):
                    field = "files" if endpoint == "/detect-batch" else "file"
                    with patch.object(self.service.model, "predict") as predict:
                        response = self.client.post(
                            endpoint, params={"vertical_center_weight": weight},
                            files=[(field, ("parking.png", self.image_bytes, "image/png"))],
                        )
                    self.assertEqual(response.status_code, 422, response.text)
                    predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
