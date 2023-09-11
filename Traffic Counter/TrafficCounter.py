import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#Video Input
cap = cv2.VideoCapture("../Videos/trafficsmall.mp4")  # For Video
#For Webcam
#cap = cv2.VideoCapture(0)
#cap.set(3,1280)
#cap.set(4,720)

#Loading YOLO model v8
model = YOLO("../Yolo-Weights/yolov8m.pt")

#Classes that are being detected
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#Masking for excluding unimportant areas
#mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [240, 297, 673, 297]
totalCount = []

while True:
    success, img = cap.read()
    #Masking for better results
    #imgRegion = cv2.bitwise_and(img, mask)

    ##imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    ##img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(img, stream=True)

    #List of Detections
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            #currentClass will give the class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            #if confidence is low then dont detect
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:

                #Printing Current object class and confidence
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)

                #If the condition is satisfied, add it to detections array {vertical stack since its numpy array}
                currentArray = np.array([x1, y1, x2, y2, conf])
                #Stacking old detections and currentArray
                detections = np.vstack((detections, currentArray))
    #tracker.update returns the similar array with an Object ID at the end
    resultsTracker = tracker.update(detections)

    #cv2.line(Image, pt1, pt2,color of line,thickness)
    cv2.line(img, (limits[0], limits[3]), (limits[2], limits[3]), (0, 0, 255), 4)

    for result in resultsTracker:
        #assigning the id
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        # Width = x2-x1;Height = y2-y1;x1,x2,y1,y2 - points
        w, h = x2 - x1, y2 - y1

        #Corners of vehicle detecting box
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

        #Vehicle id
        cvzone.putTextRect(img, f'vid: {int(id)}', (max(0, x1), max(35, y1)),scale=1.5, thickness=2, colorT=(0, 0, 0), colorR=(240, 255, 255), offset=5)

        #Point at the centre of a vehicle for easy counting
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 3, (255, 255, 0), cv2.FILLED)

        #increasing the count when a specific region crosses the line
        if limits[0] < cx < limits[2] and limits[1] - 5 < cy < limits[1] + 5:
            #if id is not present in totalCount then append
            if totalCount.count(id) == 0:
                totalCount.append(id)

                #Printing green colour when a vehicle crosses the line
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    #Vehicles Counted
    vehicle_count = str(len(totalCount))

    #Creating a box and displaying Traffic Count
    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cvzone.putTextRect(img,f'Traffic Count: {str(len(totalCount))}',(25,50),scale=3,thickness=2,colorT=(75, 0, 130),
                colorR=(230, 230, 250),font=cv2.FONT_HERSHEY_PLAIN,offset=10,border=2,colorB=(0, 0, 0))

    #Using File Handling to store the vehicle count in a file.
    f=open("../vehicle_count.txt", "w")
    f.write("Traffic Count is ")
    f.write(vehicle_count)
    f.close()

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)

