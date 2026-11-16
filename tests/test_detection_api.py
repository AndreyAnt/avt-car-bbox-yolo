import importlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def _result(boxes_xyxy, class_ids=None):
    boxes = MagicMock()
    boxes.__len__.return_value = len(boxes_xyxy)
    boxes.xyxy.cpu.return_value.numpy.return_value = np.asarray(boxes_xyxy, dtype=float)
    boxes.conf.cpu.return_value.numpy.return_value = np.full(len(boxes_xyxy), 0.8)
    boxes.cls.cpu.return_value.numpy.return_value = np.asarray(
        class_ids if class_ids is not None else [2] * len(boxes_xyxy), dtype=int,
    )
    return SimpleNamespace(boxes=boxes)


class DetectionAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep real request parsing, image decoding and selection, but avoid
        # loading model weights or running inference in deterministic API tests.
        fake_yolo = SimpleNamespace(YOLO=Mock(return_value=Mock(
            names={0: "person", 2: "car", 5: "bus", 7: "truck", 8: "boat"},
        )))
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
        for call in predict.call_args_list:
            self.assertEqual(call.kwargs["classes"], [2, 5, 7])
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

    def test_truck_and_bus_candidates_are_enabled_without_request_flags(self):
        target = [500, 200, 1300, 800]
        background_car = [100, 400, 300, 600]
        for endpoint in ("/detect", "/detect-batch"):
            for class_id in (5, 7):
                with self.subTest(endpoint=endpoint, class_id=class_id):
                    def predict(*, classes, **kwargs):
                        # Emulate detector filtering: a car-only regression
                        # would leave only the smaller background car here.
                        detections = [(target, class_id), (background_car, 2)]
                        kept = [(box, cls) for box, cls in detections if cls in classes]
                        return [_result([box for box, _ in kept], [cls for _, cls in kept])]
                    field = "files" if endpoint == "/detect-batch" else "file"
                    with patch.object(self.service.model, "predict", side_effect=predict):
                        response = self.client.post(
                            endpoint, files=[(field, ("vehicle.png", self.image_bytes, "image/png"))],
                        )
                    self.assertEqual(response.status_code, 200, response.text)
                    payload = response.json()
                    if endpoint == "/detect-batch":
                        payload = payload["items"][0]
                    self.assertEqual(payload["bbox_xyxy"], target)

    def test_seam_verification_accepts_a_vehicle_reclassified_as_truck(self):
        edge = [0, 400, 300, 600]
        rolled = [700, 400, 1300, 600]
        for endpoint in ("/detect", "/detect-batch"):
            with self.subTest(endpoint=endpoint):
                payload = self._post(endpoint, [[_result([edge], [2])], [_result([rolled], [7])]], {})
                self.assertEqual(payload["bbox_xyxy"], [1700, 400, 300, 600])
                self.assertTrue(payload["wraps_x"])

    def test_candidate_labels_preserve_detected_vehicle_class(self):
        for class_id, label in ((2, "car"), (5, "bus"), (7, "truck")):
            with self.subTest(class_id=class_id):
                _, candidates = self.service._result_to_payload(
                    _result([[500, 200, 1300, 800]], [class_id]), 2000, 1000,
                )
                self.assertEqual(candidates[0]["label"], label)

    def test_health_reports_vehicle_classes_and_preserves_legacy_alias(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["vehicle_class_ids"], [2, 5, 7])
        self.assertEqual(response.json()["car_class_ids"], [2, 5, 7])

    def test_warm_up_uses_all_supported_vehicle_classes(self):
        was_ready = self.service.app.state.model_ready
        try:
            with patch.object(self.service.model, "predict") as predict:
                self.service.warm_up_model()
            self.assertEqual(predict.call_args.kwargs["classes"], [2, 5, 7])
        finally:
            self.service.app.state.model_ready = was_ready


if __name__ == "__main__":
    unittest.main()
