from typing import Any, Dict, List, Optional, Sequence, Tuple


SEAM_EDGE_FRACTION = 0.03
SEAM_MIN_EDGE_AREA_FRACTION = 0.004
SEAM_MIN_HORIZONTAL_OVERLAP = 0.35
SEAM_MIN_VERTICAL_OVERLAP = 0.45


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bbox_metrics(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> Dict[str, Any]:
    """Calculate normalized bbox geometry for a panorama image.

    Inputs are an `xyxy` bbox and the source image size. X and Y coordinates are
    clamped into the image bounds before measurement. If `x2 < x1`, the bbox is
    treated as crossing the horizontal panorama seam, so width and area are
    measured as the visible right-edge segment plus the visible left-edge
    segment.

    Returns a dict with pixel `area`, `width`, `height`, `center_xy`, and a
    `wraps_x` flag.
    """
    x1c = _clamp(x1, 0.0, float(img_w))
    x2c = _clamp(x2, 0.0, float(img_w))
    y1c = _clamp(y1, 0.0, float(img_h))
    y2c = _clamp(y2, 0.0, float(img_h))

    wraps_x = x2c < x1c
    if wraps_x:
        width = max(0.0, float(img_w) - x1c) + max(0.0, x2c)
        cx = (x1c + width / 2.0) % float(img_w) if img_w > 0 else 0.0
    else:
        width = max(0.0, x2c - x1c)
        cx = (x1c + x2c) / 2.0

    height = max(0.0, y2c - y1c)
    cy = (y1c + y2c) / 2.0
    area = width * height

    return {
        "area": area,
        "width": width,
        "height": height,
        "center_xy": [cx, cy],
        "wraps_x": wraps_x,
    }


def _rank_car_candidates_xyxy(
    boxes_xyxy: Sequence[Sequence[float]],
    confs: Sequence[float],
    img_w: int,
    img_h: int,
) -> List[Tuple[int, float]]:
    """Rank YOLO car detections by likely main-car priority.

    Inputs are parallel sequences of YOLO `xyxy` boxes and confidence values,
    plus the source image size used to measure wrapped and non-wrapped box area.

    Returns `(box_index, area_score)` pairs sorted descending. Visible area is
    the primary ranking signal; confidence only breaks exact area ties.
    """
    ranked = []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        metrics = _bbox_metrics(
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            img_w,
            img_h,
        )
        ranked.append((i, float(metrics["area"])))

    return sorted(
        ranked,
        key=lambda item: (item[1], float(confs[item[0]])),
        reverse=True,
    )


def _pick_main_car_xyxy(
    boxes_xyxy: Sequence[Sequence[float]],
    confs: Sequence[float],
    img_w: int,
    img_h: int,
) -> Tuple[int, float]:
    """Return the top-ranked car index and its confidence.

    Inputs match `_rank_car_candidates_xyxy`. The output is the selected index
    into the original YOLO box array and that box's confidence value.
    """
    ranked = _rank_car_candidates_xyxy(boxes_xyxy, confs, img_w, img_h)
    best_i = ranked[0][0]
    return best_i, float(confs[best_i])


def _vertical_overlap_ratio(a_bbox, b_bbox) -> float:
    """Measure vertical overlap between two bboxes.

    Inputs are two `xyxy` bboxes. The return value is overlap height divided by
    the smaller bbox height, so `1.0` means the shorter vertical span is fully
    covered and `0.0` means no vertical overlap.
    """
    a_y1, a_y2 = float(a_bbox[1]), float(a_bbox[3])
    b_y1, b_y2 = float(b_bbox[1]), float(b_bbox[3])
    overlap = max(0.0, min(a_y2, b_y2) - max(a_y1, b_y1))
    min_height = min(max(0.0, a_y2 - a_y1), max(0.0, b_y2 - b_y1))
    if min_height <= 0.0:
        return 0.0
    return overlap / min_height


def _horizontal_segments(bbox, img_w: int) -> List[Tuple[float, float]]:
    """Convert a bbox into one or two non-wrapped horizontal image segments.

    Inputs are an `xyxy` bbox and image width. A normal bbox returns one
    `(start_x, end_x)` segment. A seam-wrapped bbox, where `x2 < x1`, returns
    two segments: right edge to image end, and image start to left edge.
    """
    x1 = _clamp(float(bbox[0]), 0.0, float(img_w))
    x2 = _clamp(float(bbox[2]), 0.0, float(img_w))

    if x2 < x1:
        return [(x1, float(img_w)), (0.0, x2)]
    return [(x1, x2)]


def _horizontal_overlap_ratio(a_bbox, b_bbox, img_w: int) -> float:
    """Measure horizontal overlap between two bboxes, including seam wrapping.

    Inputs are two `xyxy` bboxes and image width. The return value is overlap
    width divided by the smaller bbox width after both boxes are split into
    non-wrapped horizontal segments.
    """
    overlap = 0.0
    for a_start, a_end in _horizontal_segments(a_bbox, img_w):
        for b_start, b_end in _horizontal_segments(b_bbox, img_w):
            overlap += max(0.0, min(a_end, b_end) - max(a_start, b_start))

    a_width = sum(end - start for start, end in _horizontal_segments(a_bbox, img_w))
    b_width = sum(end - start for start, end in _horizontal_segments(b_bbox, img_w))
    min_width = min(a_width, b_width)
    if min_width <= 0.0:
        return 0.0
    return overlap / min_width


def _seam_edge_px(img_w: int) -> float:
    return max(24.0, img_w * SEAM_EDGE_FRACTION)


def _seam_min_edge_area(img_w: int, img_h: int) -> float:
    return img_w * img_h * SEAM_MIN_EDGE_AREA_FRACTION


def _edge_sides(candidate: Dict[str, Any], img_w: int) -> List[str]:
    """Return which panorama seam edges a candidate touches.

    Input is a candidate payload with `bbox_xyxy` plus image width. The output
    contains `"left"`, `"right"`, both, or neither, based on the configured seam
    edge threshold.
    """
    edge_px = _seam_edge_px(img_w)
    x1, _, x2, _ = candidate["bbox_xyxy"]
    sides = []
    if x1 <= edge_px:
        sides.append("left")
    if x2 >= img_w - edge_px:
        sides.append("right")
    return sides


def _find_seam_edge_candidates(
    candidates: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """Filter first-pass candidates that can seed roll verification.

    Inputs are ranked candidate payloads and image size. A candidate is kept
    when it is large enough to be meaningful and touches the left or right
    panorama edge.

    Returns the original candidate payloads that should trigger a rolled
    inference pass.
    """
    min_area = _seam_min_edge_area(img_w, img_h)
    return [
        candidate
        for candidate in candidates
        if candidate["area"] >= min_area and _edge_sides(candidate, img_w)
    ]


def _rolled_candidate_match_score(
    mapped_bbox,
    mapped_area: float,
    edge_candidates: List[Dict[str, Any]],
    img_w: int,
) -> Optional[Tuple[int, float]]:
    """Score how well a rolled-pass bbox confirms first-pass edge fragments.

    Inputs are a rolled-pass bbox already mapped back to original image
    coordinates, its mapped area, first-pass seam-edge candidates, and image
    width. A match requires enough vertical and horizontal overlap, and the
    mapped full bbox must be larger than the edge fragment it confirms.

    Returns `(matched_count, match_score)` when at least one edge fragment is
    confirmed, otherwise `None`.
    """
    matched_count = 0
    match_score = 0.0

    for edge_candidate in edge_candidates:
        vertical_overlap = _vertical_overlap_ratio(
            mapped_bbox,
            edge_candidate["bbox_xyxy"],
        )
        if vertical_overlap < SEAM_MIN_VERTICAL_OVERLAP:
            continue

        horizontal_overlap = _horizontal_overlap_ratio(
            mapped_bbox,
            edge_candidate["bbox_xyxy"],
            img_w,
        )
        if horizontal_overlap < SEAM_MIN_HORIZONTAL_OVERLAP:
            continue

        if mapped_area <= edge_candidate["area"]:
            continue

        matched_count += 1
        match_score += edge_candidate["area"] * vertical_overlap * horizontal_overlap

    if matched_count == 0:
        return None

    return matched_count, match_score


def _map_rolled_bbox_to_original(bbox_xyxy, shift: int, img_w: int) -> List[float]:
    """Map a bbox from rolled image coordinates back to original coordinates.

    Inputs are an `xyxy` bbox detected on an image rolled horizontally by
    `shift` pixels and the original image width. Each x coordinate is shifted
    back with modulo arithmetic, which may intentionally produce `x2 < x1` for
    seam-wrapped boxes.

    Returns the mapped `xyxy` bbox in the original image coordinate system.
    """
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    return [
        (x1 - shift) % img_w,
        y1,
        (x2 - shift) % img_w,
        y2,
    ]


def _candidate_payload(
    rank: int,
    index: int,
    xyxy: Sequence[float],
    conf: float,
    score: float,
    img_w: int,
    img_h: int,
) -> Dict[str, Any]:
    """Build the API/debug payload for one car candidate.

    Inputs are the candidate rank, original YOLO detection index, `xyxy` bbox,
    YOLO confidence, ranking score, and source image size.

    Returns a dict containing identity fields, bbox, confidence, label, score,
    and `_bbox_metrics` output. Wrapped candidates also include explicit
    `wrapped_width` and `wrapped_area` aliases for inspection.
    """
    x1, y1, x2, y2 = map(float, xyxy)
    metrics = _bbox_metrics(x1, y1, x2, y2, img_w, img_h)
    payload = {
        "rank": rank,
        "index": int(index),
        "bbox_xyxy": [x1, y1, x2, y2],
        "confidence": float(conf),
        "score": score,
        "label": "car",
        **metrics,
    }
    if metrics["wraps_x"]:
        payload["wrapped_width"] = metrics["width"]
        payload["wrapped_area"] = metrics["area"]
    return payload


def _select_verified_rolled_candidate(
    rolled_candidates: List[Dict[str, Any]],
    edge_candidates: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    shift: int,
) -> Optional[Dict[str, Any]]:
    """Pick the best rolled-pass detection that confirms a seam-wrapped car.

    Inputs are candidate payloads from the rolled inference pass, first-pass
    seam-edge candidates, source image size, and the horizontal roll shift. Each
    rolled bbox is mapped back to original coordinates, kept only if it becomes
    `wraps_x=True`, then scored against the first-pass edge candidates.

    Returns a compact detection payload with `bbox_xyxy`, `confidence`, and
    `wraps_x=True` for the best verified wrapped car. Returns `None` when no
    rolled candidate confirms the first-pass edge evidence.
    """
    best_payload = None
    best_score: Optional[Tuple[int, float, float, float]] = None

    for rolled_candidate in rolled_candidates:
        mapped_bbox = _map_rolled_bbox_to_original(
            rolled_candidate["bbox_xyxy"],
            shift,
            img_w,
        )
        mapped_metrics = _bbox_metrics(*mapped_bbox, img_w, img_h)
        if not mapped_metrics["wraps_x"]:
            continue

        match = _rolled_candidate_match_score(
            mapped_bbox,
            mapped_metrics["area"],
            edge_candidates,
            img_w,
        )
        if match is None:
            continue

        matched_count, match_score = match
        score = (
            matched_count,
            match_score,
            mapped_metrics["area"],
            float(rolled_candidate["confidence"]),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_payload = {
                "bbox_xyxy": mapped_bbox,
                "confidence": rolled_candidate["confidence"],
                "wraps_x": True,
            }

    return best_payload
