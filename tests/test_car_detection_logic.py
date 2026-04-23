import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from car_detection_logic import (
    _bbox_metrics,
    _candidate_payload,
    _find_seam_edge_candidates,
    _map_rolled_bbox_to_original,
    _select_verified_rolled_candidate,
)


IMG_W = 6080
IMG_H = 3040
SHIFT = IMG_W // 2


@dataclass(frozen=True)
class RunpodCase:
    pack: str
    name: str
    bbox_xyxy: List[float]
    confidence: float
    wraps_x: bool


REAL_RUNPOD_CASES = (
    RunpodCase("b_pack_v2", "b01", [4768.57177734375, 1406.0093994140625, 5718.76708984375, 2030.904296875], 0.7518571019172668, False),
    RunpodCase("b_pack_v2", "b02", [5041.008056640625, 1356.79443359375, 321.832275390625, 2081.690673828125], 0.8723628520965576, True),
    RunpodCase("b_pack_v2", "b03", [5209.2158203125, 1325.1207275390625, 628.794677734375, 2004.32666015625], 0.9122459888458252, True),
    RunpodCase("b_pack_v2", "b04", [5503.612548828125, 1325.8834228515625, 850.34619140625, 2058.998291015625], 0.8547395467758179, True),
    RunpodCase("b_pack_v2", "b05", [5720.19921875, 1327.997314453125, 898.320068359375, 2062.688720703125], 0.7769997119903564, True),
    RunpodCase("b_pack_v2", "b08", [4821.11572265625, 1334.7613525390625, 5740.7275390625, 1958.142578125], 0.6050910353660583, False),
    RunpodCase("b_pack_v2", "b09", [5055.3763427734375, 1315.33740234375, 265.34814453125, 2056.953857421875], 0.8603984713554382, True),
    RunpodCase("b_pack_v2", "b10", [5147.520263671875, 1304.89892578125, 548.54833984375, 2043.3975830078125], 0.8884391784667969, True),
    RunpodCase("b_pack_v2", "b11", [5340.797607421875, 1315.631591796875, 768.81103515625, 1994.4276123046875], 0.9047341346740723, True),
    RunpodCase("b_pack_v2", "b12", [5744.67041015625, 1374.087158203125, 990.14990234375, 2065.544921875], 0.8516954183578491, True),
    RunpodCase("b_pack_v2", "b13", [400.55133056640625, 1403.00830078125, 1344.2933349609375, 2028.29638671875], 0.865325927734375, False),
    RunpodCase("b_pack_v2", "b14", [5633.31689453125, 1440.7786865234375, 314.432373046875, 2041.154541015625], 0.8510506749153137, True),
    RunpodCase("c_pack_v2", "c01", [4673.95458984375, 1568.0244140625, 5626.083984375, 2237.002685546875], 0.8011998534202576, False),
    RunpodCase("c_pack_v2", "c02", [5010.817138671875, 1584.992431640625, 283.3779296875, 2287.7080078125], 0.728740930557251, True),
    RunpodCase("c_pack_v2", "c03", [5295.505859375, 1742.979248046875, 741.45947265625, 2338.988037109375], 0.9158914685249329, True),
    RunpodCase("c_pack_v2", "c04", [5529.30615234375, 1747.4122314453125, 902.39794921875, 2383.955078125], 0.9434539675712585, True),
    RunpodCase("c_pack_v2", "c05", [5865.37158203125, 1706.6988525390625, 1053.5263671875, 2391.1806640625], 0.8726948499679565, True),
    RunpodCase("c_pack_v2", "c07", [5741.097412109375, 1701.5555419921875, 380.25048828125, 2303.9423828125], 0.38059571385383606, True),
    RunpodCase("c_pack_v2", "c08", [4940.79541015625, 1608.1929931640625, 5864.94873046875, 2307.62158203125], 0.40187644958496094, False),
    RunpodCase("c_pack_v2", "c09", [5202.410888671875, 1720.674072265625, 414.726318359375, 2390.49365234375], 0.8325822949409485, True),
    RunpodCase("c_pack_v2", "c10", [5262.876708984375, 1722.9691162109375, 647.543701171875, 2373.553955078125], 0.8547877073287964, True),
    RunpodCase("c_pack_v2", "c11", [5427.05419921875, 1761.8055419921875, 848.634033203125, 2330.337890625], 0.8943361043930054, True),
    RunpodCase("c_pack_v2", "c12", [5728.10693359375, 1581.5477294921875, 989.900634765625, 2288.6162109375], 0.795332133769989, True),
    RunpodCase("c_pack_v2", "c13", [299.9996032714844, 1584.3597412109375, 1273.2659912109375, 2237.14697265625], 0.7520710229873657, False),
    RunpodCase("c_pack_v2", "c14", [5659.81005859375, 1588.2137451171875, 339.468994140625, 2252.620849609375], 0.8094831109046936, True),
)


def _assert_bbox_almost_equal(test_case, actual, expected, places=4):
    test_case.assertEqual(len(actual), len(expected))
    for actual_value, expected_value in zip(actual, expected):
        test_case.assertAlmostEqual(actual_value, expected_value, places=places)


def _cases(pack: str, wraps_x=None) -> Iterable[RunpodCase]:
    for case in REAL_RUNPOD_CASES:
        if case.pack != pack:
            continue
        if wraps_x is not None and case.wraps_x != wraps_x:
            continue
        yield case


def _rolled_bbox_for(bbox_xyxy: List[float]) -> List[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [
        (x1 + SHIFT) % IMG_W,
        y1,
        (x2 + SHIFT) % IMG_W,
        y2,
    ]


def _left_edge_seed(case: RunpodCase):
    x2 = case.bbox_xyxy[2]
    _, y1, _, y2 = case.bbox_xyxy
    return _candidate_payload(
        1,
        0,
        [0.0, y1, x2, y2],
        case.confidence,
        0.0,
        IMG_W,
        IMG_H,
    )


def _right_edge_seed(case: RunpodCase):
    x1 = case.bbox_xyxy[0]
    _, y1, _, y2 = case.bbox_xyxy
    return _candidate_payload(
        1,
        0,
        [x1, y1, float(IMG_W), y2],
        case.confidence,
        0.0,
        IMG_W,
        IMG_H,
    )


def _rolled_candidate(case: RunpodCase):
    return _candidate_payload(
        1,
        0,
        _rolled_bbox_for(case.bbox_xyxy),
        case.confidence,
        0.0,
        IMG_W,
        IMG_H,
    )


class CarDetectionLogicTests(unittest.TestCase):
    def test_fixture_set_uses_selected_real_runpod_outputs(self):
        names = [case.name for case in REAL_RUNPOD_CASES]

        self.assertEqual(
            names,
            [
                "b01",
                "b02",
                "b03",
                "b04",
                "b05",
                "b08",
                "b09",
                "b10",
                "b11",
                "b12",
                "b13",
                "b14",
                "c01",
                "c02",
                "c03",
                "c04",
                "c05",
                "c07",
                "c08",
                "c09",
                "c10",
                "c11",
                "c12",
                "c13",
                "c14",
            ],
        )
        self.assertNotIn("b06", names)
        self.assertNotIn("b07", names)
        self.assertNotIn("c06", names)

    def test_embedded_fixtures_match_local_runpod_json_when_available(self):
        output_root = Path(__file__).resolve().parents[1] / "examples" / "runpod_outputs"
        if not output_root.exists():
            self.skipTest("examples/runpod_outputs is not available")

        for case in REAL_RUNPOD_CASES:
            with self.subTest(case=case.name):
                response_path = output_root / case.pack / f"{case.name}_response.json"
                with response_path.open() as response_file:
                    response = json.load(response_file)

                self.assertEqual(response["image_size"], {"w": IMG_W, "h": IMG_H})
                self.assertEqual(response["wraps_x"], case.wraps_x)
                self.assertAlmostEqual(response["confidence"], case.confidence, places=6)
                _assert_bbox_almost_equal(self, response["bbox_xyxy"], case.bbox_xyxy)

    def test_real_runpod_bbox_metrics_preserve_wrap_flags(self):
        for case in REAL_RUNPOD_CASES:
            with self.subTest(case=case.name):
                metrics = _bbox_metrics(*case.bbox_xyxy, IMG_W, IMG_H)

                self.assertEqual(metrics["wraps_x"], case.wraps_x)
                self.assertGreater(metrics["area"], 0.0)
                if case.wraps_x:
                    self.assertLess(case.bbox_xyxy[2], case.bbox_xyxy[0])
                    self.assertIn("b" if case.pack.startswith("b") else "c", case.name)
                else:
                    self.assertGreaterEqual(case.bbox_xyxy[2], case.bbox_xyxy[0])

    def test_edge_seed_finder_keeps_only_real_edge_fragments(self):
        left_seed = _left_edge_seed(next(_cases("b_pack_v2", wraps_x=True)))
        right_seed = _right_edge_seed(next(_cases("c_pack_v2", wraps_x=True)))
        interior_case = next(_cases("b_pack_v2", wraps_x=False))
        interior = _candidate_payload(
            3,
            2,
            interior_case.bbox_xyxy,
            interior_case.confidence,
            0.0,
            IMG_W,
            IMG_H,
        )

        edge_candidates = _find_seam_edge_candidates(
            [left_seed, right_seed, interior],
            IMG_W,
            IMG_H,
        )

        self.assertEqual(edge_candidates, [left_seed, right_seed])

    def test_real_wrapped_b_pack_outputs_round_trip_through_rolled_coordinates(self):
        for case in _cases("b_pack_v2", wraps_x=True):
            with self.subTest(case=case.name):
                rolled_bbox = _rolled_bbox_for(case.bbox_xyxy)
                self.assertLess(rolled_bbox[0], rolled_bbox[2])

                mapped_bbox = _map_rolled_bbox_to_original(rolled_bbox, SHIFT, IMG_W)
                _assert_bbox_almost_equal(self, mapped_bbox, case.bbox_xyxy)

    def test_real_wrapped_c_pack_outputs_round_trip_through_rolled_coordinates(self):
        for case in _cases("c_pack_v2", wraps_x=True):
            with self.subTest(case=case.name):
                rolled_bbox = _rolled_bbox_for(case.bbox_xyxy)
                self.assertLess(rolled_bbox[0], rolled_bbox[2])

                mapped_bbox = _map_rolled_bbox_to_original(rolled_bbox, SHIFT, IMG_W)
                _assert_bbox_almost_equal(self, mapped_bbox, case.bbox_xyxy)

    def test_roll_verify_accepts_real_wrapped_b_pack_outputs_from_left_edge_seed(self):
        for case in _cases("b_pack_v2", wraps_x=True):
            with self.subTest(case=case.name):
                verified = _select_verified_rolled_candidate(
                    [_rolled_candidate(case)],
                    [_left_edge_seed(case)],
                    IMG_W,
                    IMG_H,
                    SHIFT,
                )

                self.assertIsNotNone(verified)
                self.assertTrue(verified["wraps_x"])
                self.assertAlmostEqual(verified["confidence"], case.confidence, places=6)
                _assert_bbox_almost_equal(self, verified["bbox_xyxy"], case.bbox_xyxy)

    def test_roll_verify_accepts_real_wrapped_c_pack_outputs_from_right_edge_seed(self):
        for case in _cases("c_pack_v2", wraps_x=True):
            with self.subTest(case=case.name):
                verified = _select_verified_rolled_candidate(
                    [_rolled_candidate(case)],
                    [_right_edge_seed(case)],
                    IMG_W,
                    IMG_H,
                    SHIFT,
                )

                self.assertIsNotNone(verified)
                self.assertTrue(verified["wraps_x"])
                self.assertAlmostEqual(verified["confidence"], case.confidence, places=6)
                _assert_bbox_almost_equal(self, verified["bbox_xyxy"], case.bbox_xyxy)

    def test_roll_verify_rejects_real_non_wrapped_outputs(self):
        for case in [*list(_cases("b_pack_v2", wraps_x=False)), *list(_cases("c_pack_v2", wraps_x=False))]:
            with self.subTest(case=case.name):
                verified = _select_verified_rolled_candidate(
                    [_rolled_candidate(case)],
                    [_candidate_payload(1, 0, [0.0, case.bbox_xyxy[1], 500.0, case.bbox_xyxy[3]], case.confidence, 0.0, IMG_W, IMG_H)],
                    IMG_W,
                    IMG_H,
                    SHIFT,
                )

                self.assertIsNone(verified)


if __name__ == "__main__":
    unittest.main()
