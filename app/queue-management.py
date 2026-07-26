import cv2
from ultralytics import solutions
import yaml

def queue_management():
    # Read video file
    with open("app/queue.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    cap = cv2.VideoCapture(config["CAP"])
    assert cap.isOpened(), "Error reading video file"

    # Video writer
    w, h, fps = (int(cap.get(x)) 
                for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS))
    
    video_writer = cv2.VideoWriter(config["VIDEO_WRITER"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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
    
        video_writer.write(results.plot_im)  # write the processed frame.

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()  # destroy all opened windows

if __name__ == "__main__":
    queue_management()
