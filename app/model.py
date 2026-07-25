from ultralytics import YOLO
import cv2
def model_creation(y,x,w,h):

    model = YOLO('yolo12n.pt')
    cap = cv2.VideoCapture('tests/video2.mp4')
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("no video")
            break
        frame = frame[y:y+h, x:x+w]
        #results = model.track(source= frame, vid_stride = 2, classes = [0], conf=0.35, show=True, save=True, persist=True, iou=0.7, tracker = "app/trackers/botsort.yaml")
                                            # stream=True , visualize = True
        cv2.imshow('Video Playback', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
            
        


    """for result in results:  
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box
        print(names, confs)"""

model_creation(20,20,1000,400)