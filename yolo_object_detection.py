import cv2
import numpy as np

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Load Yolo
#net = cv2.dnn.readNet("yolov4-custom_final.weights", "yolov4-custom.cfg")
#net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#net = cv2.dnn.readNet("yolov3_custom_6000.weights", "yolov3_v1_custom.cfg")
net = cv2.dnn.readNet('yolov3_custom_2000_v3.weights', 'yolov3_v1_custom.cfg')
classes = []
with open("classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#colors = np.random.uniform(0, 255, size=(len(classes), 3))
color = [0, 255, 0]

while True:
    ret, img = cap.read()
    #img = cv2.imread("room_ser.jpg")
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)
    #print(len(boxes))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cx = int(w/2 + x)
            cy = int(h/2 + y)
            #color = colors[class_ids[i]]
            cv2.circle(img, (cx, cy), 4, color, 6)
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x + 30, y + 30), font, 0.5, color, 1)
            cv2.putText(img, f"{len(indexes)}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            #print(len(indexes))

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27: break

cv2.destroyAllWindows()