# Queue Analysis

<!-- One or two sentences: what this project does and why. -->

## Overview

<!-- What problem this solves, who it's for, high-level approach. -->

## Architecture

<!--
Record-then-batch-process design: cameras record to files continuously (cheap,
no GPU needed for capture); a separate step runs YOLO tracking + ROI/queue
analysis on recorded files on demand, so GPU only spins up when processing is
actually triggered, not 24/7.
-->

## Progress So Far

- Detection + tracking via `ultralytics.solutions.QueueManager` (YOLO26 + BoT-SORT), replacing plain per-frame detection with persistent track IDs across frames.
- ROI ("queue region") defined as a polygon and passed to `QueueManager` directly, rather than cropping the frame manually.
- Pipeline is config-driven via `app/queue.yaml` (video source, model, tracker config, region points, conf/iou thresholds, output path) instead of hardcoded values or env vars.
- `app/queue-management.py` split into single-responsibility functions: `load_config`, `cap_check` / `release_cap`, `video_writer`, `model_creator`, `video_processor`, orchestrated by `queue_management()`.
- Custom tracker configs in `app/trackers/` (`botsort.yaml`, `bytetrack.yaml`) tuned for queue scenarios (occlusion handling via `track_buffer`, ReID for dense/occluding crowds).
- `Dockerfile` using a CUDA/PyTorch-ready base image (`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`) with `.dockerignore` excluding `venv/`, `.git/`, `.env`, `runs/`, `docs/`, `tests/`.
- `docker-compose.yml` in progress — GPU device reservation + volume mounts for input video, output, and config.

**Next up:** ROI-filtered per-frame logging in `video_processor` (ties each `track_id` to whether it's inside the queue region) to observe real entry/exit/occlusion behavior before finalizing storage granularity; then a two-service persistence layer — see Roadmap.

## Setup

<!-- venv/dependency install steps, GPU/driver prerequisites, .env / queue.yaml setup. -->

## Usage

<!-- How to run app/queue-management.py, how to run via Docker, expected inputs/outputs. -->

## Project Structure

<!-- Brief map of app/, tests/, app/trackers/, queue.yaml, Dockerfile, docker-compose.yml. -->

## Roadmap

- Decide storage granularity (interval-sampled snapshots vs on-change vs per-track dwell time) from observed ROI entry/exit logs.
- **Two-service persistence layer** (separate from the RunPod deployment):
  - `processor` service — runs the queue-analysis pipeline (or consumes RunPod job results) and POSTs results over HTTP; never touches SQLite directly.
  - `storage-api` service — small Flask/FastAPI app that exclusively owns the SQLite file, exposing routes to insert and query records. Sole owner avoids SQLite's multi-writer/locking issues across processes.
  - Wired together in `docker-compose.yml`: shared internal network for service-to-service calls, named volume so the SQLite file persists across restarts.
- RTSP/continuous recorder as the "record" half of the record-then-batch architecture.
- Batch trigger/API to kick off processing on demand (RunPod Serverless — done for the GPU/tracking side; still need the local trigger + result-persistence loop).
- Multi-camera support.

## License

<!-- TBD -->
