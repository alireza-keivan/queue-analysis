import cv2
from ultralytics import solutions
import yaml
import logging
import numpy as np
import uuid

def load_config(file_path="app/queue.yaml"):
    # Read video file
    with open(file_path, "r") as file:
        try: 
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            logging.error(f"Error reading YAML file: {e}")
            return None
    return config

def cap_check(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return None
    return cap

def release_cap(check):
    if check:
        check.release()
        #cv2.destroyAllWindows()
        return "cap released"
    else:
        logging.error("CAP IS NOT OPENED")
        return None

def video_writer(capture, path):
    # Create a VideoWriter object
    w, h, fps = (int(capture.get(x)) 
                    for x in (
                    cv2.CAP_PROP_FRAME_WIDTH,
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    cv2.CAP_PROP_FPS))
    
    writer = cv2.VideoWriter(path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (w, h))
    return writer    

def model_creator(config):
    # Initialize queue manager object
    queuemanager = solutions.QueueManager(
        show=False,  # display the output
        model=config["MODEL"],  # path to the YOLO26 model file
        region=config["QUEUE_REGION"],
        tracker=config["TRACKER"],
        conf=config["CONF"],
        iou=config["IOU"],
        classes=[0]
    )
    return queuemanager

def video_processor(cap, queue_manager, writer, job_id):
    # Process video
    region = np.array(queue_manager.region, dtype=np.int32)  # built once, reused every frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    open_tracks = {}       # track_id -> entry_time (video-relative seconds)
    completed_tracks = []  # [{job_id, track_id, dwell_seconds}, ...]
    snapshots = []         # [{job_id, timestamp, queue_count}, ...]
    last_snapshot_second = None

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            logging.info("Video frame is empty or processing is complete.")
            break
        results = queue_manager(im0)
        current_time = frame_idx / fps

        inside_ids = []
        for track_id, box in zip(queue_manager.track_ids, queue_manager.boxes):
            point = (float((box[0] + box[2]) / 2), float(box[3]))  # bottom-center of box
            if cv2.pointPolygonTest(region, point, False) >= 0:
                inside_ids.append(track_id)

        # Entries: newly inside the ROI, not already being tracked as "in".
        for track_id in inside_ids:
            if track_id not in open_tracks:
                open_tracks[track_id] = current_time

        # Exits: was inside, no longer is.
        for track_id in list(open_tracks.keys()):
            if track_id not in inside_ids:
                entry_time = open_tracks.pop(track_id)
                completed_tracks.append({
                    "job_id": job_id,
                    "track_id": track_id,
                    "dwell_seconds": current_time - entry_time,
                })

        # Snapshot once per whole second of video, not every frame.
        current_second = int(current_time)
        if current_second != last_snapshot_second:
            snapshots.append({
                "job_id": job_id,
                "timestamp": current_time,
                "queue_count": len(inside_ids),
            })
            last_snapshot_second = current_second

        logging.info(f"Frame {frame_idx}: inside ROI = {inside_ids}")

        writer.write(results.plot_im)  # write the processed frame.
        frame_idx += 1

    # Video ended while these were still inside the ROI - close them out
    # using the last processed frame as their exit, so they aren't silently
    # dropped from the data.
    final_time = frame_idx / fps
    for track_id, entry_time in open_tracks.items():
        completed_tracks.append({
            "job_id": job_id,
            "track_id": track_id,
            "dwell_seconds": final_time - entry_time,
        })

    return snapshots, completed_tracks

def queue_management():
    config = load_config()
    cap = cap_check(config["CAP"])
    writer = video_writer(cap, config["VIDEO_WRITER"])
    queue_manager = model_creator(config)
    job_id = str(uuid.uuid4())
    snapshots, completed_tracks = video_processor(cap, queue_manager, writer, job_id)
    logging.info("Queue management processing complete.")

    release_cap(cap)
    writer.release()
    return snapshots, completed_tracks

if __name__ == "__main__":
    queue_management()