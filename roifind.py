import cv2
import numpy as np

def roi_find(image_input):
    net = cv2.dnn.readNet("/home/serhii/Documents/PyProjects/yolo-learning/object-detection-1/yolov3.weights", "/home/serhii/Documents/PyProjects/yolo-learning/object-detection-1/yolov3.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    classes = []
    with open('/home/serhii/Documents/PyProjects/yolo-learning/object-detection-1/coco.names','r') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    img = cv2.imread(image_input)

    height, width, channel = img.shape

    # Detecting Objects
    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0, 0, 0), swapRB=True, crop=False) 
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Show information on the screen
    # Initialise variables
    class_ids = []
    confidences = [] 
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int (center_x - w/2)
                y = int (center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Label object
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range (len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + "  " + confidence, (x, y+20), font, 1, color, 2)

            #print(x, y, w, h)

    #cv2.imshow('Img', img)
    #cv2.waitKey()

    return (x, y, w, h)
