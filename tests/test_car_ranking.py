import unittest

from car_detection_logic import (
    _candidate_payload,
    _find_seam_edge_candidates,
    _pick_main_car_xyxy,
    _rank_car_candidates_xyxy,
    _select_verified_rolled_candidate,
)


IMG_W = 2000
IMG_H = 1000
SHIFT = IMG_W // 2


def _candidate(bbox, confidence=0.8):
    return _candidate_payload(1, 0, bbox, confidence, 0.0, IMG_W, IMG_H)


def _rolled_candidate(original_bbox, confidence=0.8):
    x1, y1, x2, y2 = original_bbox
    return _candidate([(x1 + SHIFT) % IMG_W, y1, (x2 + SHIFT) % IMG_W, y2], confidence)


class CarRankingTests(unittest.TestCase):
    def test_centered_car_beats_slightly_larger_foreground_car(self):
        boxes = [[0, 650, 600, 850], [700, 400, 1200, 600]]

        index, confidence = _pick_main_car_xyxy(boxes, [0.99, 0.7], IMG_W, IMG_H)

        self.assertEqual(index, 1)
        self.assertEqual(confidence, 0.7)

    def test_zero_weight_restores_area_only_ranking(self):
        boxes = [[0, 650, 600, 850], [700, 400, 1200, 600]]

        ranked = _rank_car_candidates_xyxy(boxes, [0.8, 0.8], IMG_W, IMG_H, 0.0)

        self.assertEqual(ranked, [(0, 120000.0), (1, 100000.0)])

    def test_tiny_centered_background_car_does_not_beat_large_car(self):
        boxes = [[0, 600, 800, 800], [900, 450, 1100, 550]]

        index, _ = _pick_main_car_xyxy(boxes, [0.7, 0.99], IMG_W, IMG_H)

        self.assertEqual(index, 0)

    def test_area_still_decides_when_vertical_centers_match(self):
        boxes = [[100, 400, 600, 600], [900, 400, 1500, 600]]

        index, _ = _pick_main_car_xyxy(boxes, [0.99, 0.7], IMG_W, IMG_H)

        self.assertEqual(index, 1)

    def test_confidence_breaks_equal_weighted_score_ties(self):
        boxes = [[0, 650, 800, 850], [850, 400, 1450, 600]]

        for confidences, expected in (([0.9, 0.7], 0), ([0.7, 0.9], 1)):
            with self.subTest(confidences=confidences):
                ranked = _rank_car_candidates_xyxy(boxes, confidences, IMG_W, IMG_H)
                self.assertEqual(ranked[0][1], ranked[1][1])
                self.assertEqual(ranked[0][0], expected)

    def test_horizontal_panorama_offsets_preserve_scores_and_selection(self):
        boxes = [[0, 650, 600, 850], [700, 400, 1200, 600]]
        expected = _rank_car_candidates_xyxy(boxes, [0.9, 0.7], IMG_W, IMG_H)

        for offset in (0, 137, 800, 1000, 1400, 1999):
            with self.subTest(offset=offset):
                shifted = [
                    [(x1 + offset) % IMG_W, y1, (x2 + offset) % IMG_W, y2]
                    for x1, y1, x2, y2 in boxes
                ]
                self.assertEqual(
                    _rank_car_candidates_xyxy(shifted, [0.9, 0.7], IMG_W, IMG_H),
                    expected,
                )

    def test_image_resolution_does_not_change_selection(self):
        boxes = [[0, 650, 600, 850], [700, 400, 1200, 600]]

        for scale in (0.5, 1, 2, 4):
            with self.subTest(scale=scale):
                index, _ = _pick_main_car_xyxy(
                    [[value * scale for value in box] for box in boxes],
                    [0.8, 0.8], int(IMG_W * scale), int(IMG_H * scale),
                )
                self.assertEqual(index, 1)

    def test_equal_distance_above_and_below_mid_height_has_equal_priority(self):
        boxes = [[0, 150, 600, 350], [700, 650, 1300, 850]]

        ranked = _rank_car_candidates_xyxy(boxes, [0.8, 0.8], IMG_W, IMG_H)

        self.assertEqual(ranked[0][1], ranked[1][1])

    def test_center_and_area_use_clamped_image_coordinates(self):
        boxes = [[200, -500, 800, 200], [200, 0, 800, 200]]

        ranked = _rank_car_candidates_xyxy(boxes, [0.8, 0.8], IMG_W, IMG_H)

        self.assertEqual(ranked[0][1], ranked[1][1])

    def test_empty_detections_return_empty_ranking(self):
        self.assertEqual(_rank_car_candidates_xyxy([], [], IMG_W, IMG_H), [])

    def test_zero_height_does_not_divide_by_zero(self):
        self.assertEqual(_rank_car_candidates_xyxy([[0, 0, 100, 200]], [0.8], IMG_W, 0), [(0, 0.0)])


class SeamRankingTests(unittest.TestCase):
    def test_verified_foreground_car_cannot_override_higher_priority_main_car(self):
        main = _candidate([700, 400, 1200, 600])
        rolled = _rolled_candidate([1700, 650, 300, 850])
        edges = _find_seam_edge_candidates(
            [_candidate([0, 650, 300, 850]), _candidate([1700, 650, 2000, 850])],
            IMG_W, IMG_H,
        )

        verified = _select_verified_rolled_candidate(
            [rolled], edges, IMG_W, IMG_H, SHIFT, main_candidate=main,
        )

        self.assertIsNone(verified)

    def test_small_verified_seam_car_cannot_override_large_main_car(self):
        main = _candidate([600, 200, 1400, 800])
        rolled = _rolled_candidate([1900, 500, 120, 800])
        edges = [_candidate([0, 500, 120, 800])]

        for weight in (0.0, 1.0, 4.0):
            with self.subTest(weight=weight):
                verified = _select_verified_rolled_candidate(
                    [rolled], edges, IMG_W, IMG_H, SHIFT,
                    main_candidate=main, vertical_center_weight=weight,
                )
                self.assertIsNone(verified)

    def test_zero_weight_applies_to_seam_and_main_car_comparison(self):
        main = _candidate([700, 400, 1200, 600])
        bbox = [1700, 650, 300, 850]

        verified = _select_verified_rolled_candidate(
            [_rolled_candidate(bbox)], [_candidate([0, 650, 300, 850])],
            IMG_W, IMG_H, SHIFT, main_candidate=main, vertical_center_weight=0.0,
        )

        self.assertIsNotNone(verified)
        self.assertEqual(verified["bbox_xyxy"], bbox)

    def test_verified_main_car_replaces_its_partial_edge_detection(self):
        bbox = [1700, 400, 300, 600]
        edge = _candidate([0, 400, 300, 600])

        verified = _select_verified_rolled_candidate(
            [_rolled_candidate(bbox)], [edge], IMG_W, IMG_H, SHIFT, main_candidate=edge,
        )

        self.assertIsNotNone(verified)
        self.assertEqual(verified["bbox_xyxy"], bbox)
        self.assertTrue(verified["wraps_x"])

    def test_verified_car_can_outrank_an_interior_car(self):
        bbox = [1700, 400, 300, 600]

        verified = _select_verified_rolled_candidate(
            [_rolled_candidate(bbox)], [_candidate([0, 400, 300, 600])],
            IMG_W, IMG_H, SHIFT, main_candidate=_candidate([700, 400, 1100, 600]),
        )

        self.assertIsNotNone(verified)
        self.assertEqual(verified["bbox_xyxy"], bbox)

    def test_verified_cars_compete_by_priority_instead_of_fragment_count(self):
        centered = [1700, 400, 300, 600]
        foreground = [1600, 700, 400, 900]
        edges = [
            _candidate([0, 400, 300, 600]),
            _candidate([0, 700, 400, 900]),
            _candidate([1600, 700, 2000, 900]),
        ]

        for boxes in ([foreground, centered], [centered, foreground]):
            with self.subTest(boxes=boxes):
                verified = _select_verified_rolled_candidate(
                    [_rolled_candidate(box) for box in boxes], edges, IMG_W, IMG_H, SHIFT,
                )
                self.assertIsNotNone(verified)
                self.assertEqual(verified["bbox_xyxy"], centered)

    def test_seam_score_ties_use_confidence_and_preserve_original_on_exact_tie(self):
        main = _candidate([700, 400, 1300, 600], confidence=0.8)
        bbox = [1700, 400, 300, 600]

        for confidence in (0.7, 0.8, 0.9):
            with self.subTest(confidence=confidence):
                verified = _select_verified_rolled_candidate(
                    [_rolled_candidate(bbox, confidence)], [_candidate([0, 400, 300, 600])],
                    IMG_W, IMG_H, SHIFT, main_candidate=main,
                )
                if confidence > 0.8:
                    self.assertIsNotNone(verified)
                    self.assertEqual(verified["bbox_xyxy"], bbox)
                else:
                    self.assertIsNone(verified)

    def test_high_priority_seam_car_still_requires_matching_edge_evidence(self):
        verified = _select_verified_rolled_candidate(
            [_rolled_candidate([1500, 200, 500, 600])], [_candidate([0, 700, 300, 900])],
            IMG_W, IMG_H, SHIFT,
        )

        self.assertIsNone(verified)


if __name__ == "__main__":
    unittest.main()
