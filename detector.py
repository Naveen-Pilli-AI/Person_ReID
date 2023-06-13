import cv2
import torch
from collections import defaultdict



confidence_threshold=0.5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',force_reload=True)

def object_detector(vid_frame):
    results = model(vid_frame, size=640)

    detections = results.pandas().xyxy[0]
    
    try:

        Total_people=detections['name'].value_counts()['person']
        # print(Total_people)

    except:
           pass

    person_detections = detections[detections['name'] == 'person']
    
   
    person_detections = person_detections[person_detections['confidence'] >= confidence_threshold]
    
    result_outputs=[]
    top1_person = []

    for _, detection in person_detections.iterrows():
        x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
        conf=detection['confidence']
        cls_name=detection['name']

       
        result_outputs.append({
                  cls_name: {
                        "confidence": float(conf),
                        "bounding_box": [int(x1), int(y1), int(x2), int(y2)]
                    }
                })
        

    return result_outputs