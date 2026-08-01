import cv2
from ultralytics import solutions
import yaml
import logging
import numpy as np

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

def video_processor(cap, queue_manager, writer):
    # Process video
    region = np.array(queue_manager.region, dtype=np.int32)  # built once, reused every frame
    frame_idx = 0
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            logging.info("Video frame is empty or processing is complete.")
            break
        results = queue_manager(im0)

        inside_ids = []
        for track_id, box in zip(queue_manager.track_ids, queue_manager.boxes):
            point = (float((box[0] + box[2]) / 2), float(box[3]))  # bottom-center of box
            if cv2.pointPolygonTest(region, point, False) >= 0:
                inside_ids.append(track_id)

        logging.info(f"Frame {frame_idx}: inside ROI = {inside_ids}")

        writer.write(results.plot_im)  # write the processed frame.
        frame_idx += 1
    return results

def queue_management():
    config = load_config()
    cap = cap_check(config["CAP"])
    writer = video_writer(cap, config["VIDEO_WRITER"])
    queue_manager = model_creator(config)
    result_processor = video_processor(cap, queue_manager, writer)
    logging.info("Queue management processing complete.")

    release_cap(cap)
    writer.release()
    return result_processor

if __name__ == "__main__":
    queue_management()
