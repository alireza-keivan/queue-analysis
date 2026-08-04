"""One-off diagnostic, not part of the production pipeline.

Quantifies how much ID churn is explained by real occlusion (another person's
box actually overlapping the one that vanished) versus unexplained "phantom"
switches (the ID disappears and nothing was there to cause it), and flags
candidate identity-swap moments where two tracked boxes overlap heavily.

Runs full-frame (not ROI-filtered) since the failure examples that prompted
this - a stationary person alone in frame, a crowd crossing - aren't
necessarily inside the queue region.
"""
import cv2

from app.queue_management import frame_stride


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def analyze_track_churn(cap, queue_manager, target_fps=10,
                          overlap_thresh=0.05, renumber_frames=3, renumber_iou=0.3,
                          crossing_iou=0.15):
    stride, _ = frame_stride(cap, target_fps)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(target_fps)

    # history[i] = {track_id: (x1, y1, x2, y2)} for the i-th PROCESSED frame
    history = []
    src_idx = 0
    while True:
        if not cap.grab():
            break
        pos = src_idx
        src_idx += 1
        if pos % stride != 0:
            continue
        ok, im0 = cap.retrieve()
        if not ok:
            break
        queue_manager(im0)
        history.append({
            tid: tuple(float(v) for v in box[:4])
            for tid, box in zip(queue_manager.track_ids, queue_manager.boxes)
        })

    first_seen, last_seen = {}, {}
    for i, frame in enumerate(history):
        for tid in frame:
            first_seen.setdefault(tid, i)
            last_seen[tid] = i

    last_frame_idx = len(history) - 1
    occlusion_plausible, phantom, renumber_candidates = [], [], []

    for tid, end_i in last_seen.items():
        if end_i == last_frame_idx:
            continue  # still on screen when the video ended - not a real loss
        box = history[end_i][tid]

        overlapped = any(
            other_tid != tid and _iou(box, other_box) >= overlap_thresh
            for other_tid, other_box in history[end_i].items()
        )
        (occlusion_plausible if overlapped else phantom).append(tid)

        for gap in range(1, renumber_frames + 1):
            j = end_i + gap
            if j >= len(history):
                break
            for new_tid, new_box in history[j].items():
                if first_seen.get(new_tid) == j:  # a genuinely new id
                    iou = _iou(box, new_box)
                    if iou >= renumber_iou:
                        renumber_candidates.append({
                            "old_id": tid, "new_id": new_tid,
                            "frames_gap": gap, "iou": round(iou, 3),
                            "time": round(end_i * stride / src_fps, 2),
                        })

    crossings = []
    for i, frame in enumerate(history):
        ids = list(frame.keys())
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                iou = _iou(frame[ids[a]], frame[ids[b]])
                if iou >= crossing_iou:
                    crossings.append({
                        "time": round(i * stride / src_fps, 2),
                        "id_a": ids[a], "id_b": ids[b], "iou": round(iou, 3),
                    })

    endings = len(occlusion_plausible) + len(phantom)
    return {
        "processed_frames": len(history),
        "total_distinct_ids": len(last_seen),
        "track_endings_analyzed": endings,
        "occlusion_plausible_endings": len(occlusion_plausible),
        "phantom_endings_no_overlap": len(phantom),
        "phantom_pct": round(len(phantom) / endings * 100, 1) if endings else 0.0,
        "renumber_candidates": renumber_candidates,
        "crossing_events": crossings,
    }