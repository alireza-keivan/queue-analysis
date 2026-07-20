from ultralytics import YOLO

def model_creation():

    model = YOLO('yolo12n.pt')
    #results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
    results = model('images.jpeg')
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box
        print(names, confs)