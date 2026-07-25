import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("tests/video2.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("queue_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Define queue points
queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # region points
# queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]    # polygon points

# Initialize queue manager object
queuemanager = solutions.QueueManager(
    show=True,  # display the output
    model="yolo26n.pt",  # path to the YOLO26 model file
    region=queue_region,  # pass queue region points
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    results = queuemanager(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows