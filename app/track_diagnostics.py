"""One-off diagnostic, not part of the production pipeline.

Quantifies how much ID churn is explained by real occlusion (another person's
box actually overlapping the one that vanished) versus unexplained "phantom"
switches (the ID disappears and nothing was there to cause it), and flags
candidate identity-swap moments where two tracked boxes overlap heavily.

Runs full-frame (not ROI-filtered) since the failure examples that prompted
this - a stationary person alone in frame, a crowd crossing - aren't
necessarily inside the queue region.
"""
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


def collect_track_history(cap, queue_manager, target_fps=10):
    """GPU-bound pass: runs the model over every processed frame and records
    each track's box. This is the only half of the old analyze_track_churn
    that actually needs the GPU worker - see score_track_churn for the pure
    CPU pass that used to run inline with it.

    cv2/queue_management are imported lazily here (not at module level) so
    that importing score_track_churn - the half a CPU-only worker needs -
    never pulls in cv2 or the full ultralytics/torch stack.
    """
    import cv2

    from app.queue_management import frame_stride

    stride, _ = frame_stride(cap, target_fps)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(target_fps)
    # Needed downstream to tell "walked out of shot" apart from "vanished
    # mid-frame" - box coordinates alone can't distinguish them.
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

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

    return history, stride, src_fps, frame_size


# A track whose last box sits within this many pixels of the frame border is
# treated as having walked out of shot rather than having been lost. Measured
# on the reference video, the classification is stable anywhere from 10px to
# 150px (13 of 17 no-overlap endings are frame exits at every value in that
# range), so the exact number is not load-bearing.
EDGE_MARGIN_PX = 20

# How far ahead to look for "the same person came back under a new ID".
# Counted in PROCESSED frames, so it must be read against botsort.yaml's
# track_buffer, which is the window the tracker itself keeps a lost track
# alive for (also processed frames). They were mismatched 3 vs 30 - the
# diagnostic searched 0.3s while the tracker re-associated over 3.0s, so it
# structurally could not see most of the ID switches the tracker was making.
RENUMBER_FRAMES = 30


def _touches_frame_edge(box, frame_size, margin=EDGE_MARGIN_PX):
    """True if the box is against the frame border - i.e. the person was
    leaving the shot, so losing the track there is correct behaviour."""
    if not frame_size:
        return False
    w, h = frame_size
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin


def score_track_churn(history, stride, src_fps, overlap_thresh=0.05,
                       renumber_frames=RENUMBER_FRAMES, renumber_iou=0.3,
                       crossing_iou=0.15, frame_size=None,
                       edge_margin=EDGE_MARGIN_PX):
    """Pure CPU pass over an already-collected history - no model, no cv2, no
    GPU. Safe to run anywhere: a laptop, the dashboard service, a CI box.

    frame_size ((width, height)) enables splitting no-overlap track endings
    into people who left the frame versus genuinely unexplained losses.
    Without it that split can't be computed and every no-overlap ending is
    reported as unexplained, which materially overstates tracker failure -
    on the reference video 13 of 17 were simply people walking out of shot.
    """
    first_seen, last_seen = {}, {}
    for i, frame in enumerate(history):
        for tid in frame:
            first_seen.setdefault(tid, i)
            last_seen[tid] = i

    last_frame_idx = len(history) - 1
    occlusion_plausible, phantom, renumber_candidates = [], [], []
    frame_exits, unexplained = [], []

    for tid, end_i in last_seen.items():
        if end_i == last_frame_idx:
            continue  # still on screen when the video ended - not a real loss
        box = history[end_i][tid]

        overlapped = any(
            other_tid != tid and _iou(box, other_box) >= overlap_thresh
            for other_tid, other_box in history[end_i].items()
        )
        if overlapped:
            occlusion_plausible.append(tid)
        else:
            phantom.append(tid)
            # No overlapping box explains this loss - but a person walking
            # off the edge of the frame is not a tracker failure either.
            (frame_exits if _touches_frame_edge(box, frame_size, edge_margin)
             else unexplained).append(tid)

        # First re-appearance only: one lost track can be followed by many
        # new ids over a 30-frame window, and counting all of them inflates
        # a single ID switch into several.
        for gap in range(1, renumber_frames + 1):
            j = end_i + gap
            if j >= len(history):
                break
            best = None
            for new_tid, new_box in history[j].items():
                if first_seen.get(new_tid) == j:  # a genuinely new id
                    iou = _iou(box, new_box)
                    if iou >= renumber_iou and (best is None or iou > best[1]):
                        best = (new_tid, iou)
            if best is not None:
                renumber_candidates.append({
                    "old_id": tid, "new_id": best[0],
                    "frames_gap": gap, "iou": round(best[1], 3),
                    "time": round(end_i * stride / src_fps, 2),
                })
                break

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
        # Kept for continuity: every ending with no overlapping box. This is
        # NOT the tracker-failure rate - see unexplained_* below.
        "phantom_endings_no_overlap": len(phantom),
        "phantom_pct": round(len(phantom) / endings * 100, 1) if endings else 0.0,
        # The honest split of the above. frame_exit = person left the shot
        # (correct behaviour); unexplained = genuinely lost mid-frame.
        # Both are 0 and unexplained_pct is None when frame_size is unknown,
        # since the split cannot be computed without it.
        "frame_exit_endings": len(frame_exits),
        "unexplained_endings": len(unexplained),
        "unexplained_pct": (round(len(unexplained) / endings * 100, 1)
                            if endings and frame_size else None),
        "edge_classification_available": bool(frame_size),
        "renumber_candidates": renumber_candidates,
        "renumber_window_frames": renumber_frames,
        "crossing_events": crossings,
    }


def analyze_track_churn(cap, queue_manager, target_fps=10, **score_kwargs):
    """Convenience wrapper: collect + score in one call, for local/one-off
    use. The GPU handler no longer calls this directly - it calls
    collect_track_history itself and leaves scoring to the CPU-only RunPod
    endpoint (see cpu_handler.py).
    """
    history, stride, src_fps, frame_size = collect_track_history(cap, queue_manager, target_fps)
    score_kwargs.setdefault("frame_size", frame_size)
    return score_track_churn(history, stride, src_fps, **score_kwargs)