import cv2
import numpy as np

# Load the YOLOv3 model with pre-trained weights and configuration file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg.txt')  # Fixed incorrect .cfg filename

# Load class names from the COCO dataset
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()  # Read class names into a list

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # Open default webcam (device 0)

while True:
    # Read a frame from the webcam
    ret, img = cap.read()
    if not ret:
        break  # Stop the loop if the frame is not read correctly

    height, width, _ = img.shape  # Get image dimensions

    # Convert the image into a blob for YOLO processing
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    # Set the input for the YOLO model
    net.setInput(blob)

    # Get the names of output layers in the YOLO model
    output_layers_names = net.getUnconnectedOutLayersNames()

    # Perform forward pass to get detection predictions
    layerOutputs = net.forward(output_layers_names)

    # Lists to store detected objects' information
    boxes = []        # Bounding box coordinates
    confidences = []  # Confidence scores
    class_ids = []    # Class IDs corresponding to detected objects

    # Process each output layer's detections
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]  # Extract confidence scores for all classes
            class_id = np.argmax(scores)  # Get the class with the highest confidence
            confidence = scores[class_id]  # Get the confidence of the selected class

            # Filter out weak detections with confidence threshold
            if confidence > 0.5:
                # Scale detection box coordinates back to image dimensions
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)  # Fixed: Previously, `w` was used instead of `h`

                # Store detection information
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Set font for labeling
    font = cv2.FONT_HERSHEY_PLAIN

    # Generate random colors for different classes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Draw bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Get class label
            confidence = str(round(confidences[i], 2))  # Round confidence score
            color = colors[class_ids[i]]  # Assign a unique color to the class

            # Draw a rectangle around the detected object
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Put label text with confidence score
            cv2.putText(img, label + " " + confidence, (x, y - 10), font, 2, (255, 255, 255), 2)

    # Display the image with detections
    cv2.imshow('YOLO Object Detection', img)

    # Break the loop if the 'Esc' key (27) is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
