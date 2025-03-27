import cv2
import numpy as np

# Load the pre-trained YOLO model with its weights and configuration file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg.txt')  # Ensure the correct filename extension

# Load class names from the COCO dataset
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()  # Read all class names into a list

# Open the video file (replace 'pexels.mp4' with your video file)
cap = cv2.VideoCapture('pexels.mp4')

while True:
    # Read a frame from the video
    ret, img = cap.read()
    
    # Check if the frame is read successfully
    if not ret:
        break  # Exit the loop when the video ends

    height, width, _ = img.shape  # Get the dimensions of the frame

    # Convert the image into a blob format suitable for YOLO processing
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    # Set the blob as input to the YOLO model
    net.setInput(blob)

    # Get the output layer names from the YOLO model
    output_layers_names = net.getUnconnectedOutLayersNames()

    # Perform forward propagation to get predictions
    layerOutputs = net.forward(output_layers_names)

    # Lists to store detected object data
    boxes = []        # Bounding box coordinates
    confidences = []  # Confidence scores
    class_ids = []    # Class IDs for detected objects

    # Process each detection from the output layers
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]  # Extract confidence scores for all classes
            class_id = np.argmax(scores)  # Get the class with the highest confidence
            confidence = scores[class_id]  # Get the confidence of that class

            # Apply confidence threshold (filter out weak detections)
            if confidence > 0.5:
                # Convert normalized center coordinates and box size back to original image scale
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)  # Fixed: `h` instead of `w`

                # Store bounding box information
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Define font for displaying labels
    font = cv2.FONT_HERSHEY_PLAIN

    # Generate random colors for different classes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Draw bounding boxes and labels on the detected objects
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Get object label
            confidence = str(round(confidences[i], 2))  # Format confidence score
            color = colors[class_ids[i]]  # Assign a unique color to each class

            # Draw a bounding box around the detected object
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Display the object label and confidence score
            cv2.putText(img, label + " " + confidence, (x, y - 10), font, 2, (255, 255, 255), 2)

    # Show the processed frame with detections
    cv2.imshow('YOLO Object Detection', img)

    # Exit loop if the 'Esc' key (27) is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
