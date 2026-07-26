import cv2
from ultralytics import solutions
import yaml

def load_config(file_path="app/queue.yaml"):
    # Read video file
    with open(file_path, "r") as file:
        try: 
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None
    return config

def cap_check(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    return cap

def release_cap(check):
    if check:
        check.release()
        cv2.destroyAllWindows()
        return "cap released"
    else:
        print("CAP IS NOT OPENED")
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
    
def queue_management():
    config = load_config()
    cap = cap_check(config["CAP"])
    writer = video_writer(cap, config["VIDEO_WRITER"])
    
    # Initialize queue manager object
    queuemanager = solutions.QueueManager(
        show=True,  # display the output
        model=config["MODEL"],  # path to the YOLO26 model file
        region=config["QUEUE_REGION"],
        tracker=config["TRACKER"],
        conf=config["CONF"],
        iou=config["IOU"],
        classes=[0],
        device=0,
    )
    # Process video
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = queuemanager(im0)

        print(results)  # access the output
    
        writer.write(results.plot_im)  # write the processed frame.

    release_cap(cap)
    writer.release()

#if __name__ == "__main__":
#    queue_management()
