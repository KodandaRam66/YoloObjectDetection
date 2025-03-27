# YOLOv3 Object Detection with OpenCV

This project implements real-time object detection using the YOLOv3 (You Only Look Once) model with OpenCV and Deep Neural Network (DNN) module. The program processes video frames to detect and label objects from the COCO dataset in real-time.

## Features
- Loads a pre-trained YOLOv3 model for object detection.
- Processes video files or webcam streams.
- Draws bounding boxes around detected objects with class labels and confidence scores.
- Applies Non-Maximum Suppression (NMS) to filter overlapping boxes.
- Displays the processed video output with detected objects.

## Requirements
Ensure you have the following installed:

- Python 3.x
- OpenCV
- NumPy

You can install the required dependencies using:
```sh
pip install opencv-python numpy
```

## Setup Instructions
1. Download the YOLOv3 model weights and configuration files:
   - [`yolov3.weights`](https://pjreddie.com/media/files/yolov3.weights)
   - [`yolov3.cfg`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - [`coco.names`](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

2. Place these files in the project directory.

3. Run the script with a image file:
   ```sh
   python main.py
   ```
4. Run the script with a video file:
   ```sh
   python video.py
   ```
   
   Run the script with a webcam:
   ```sh
   python cam.py
   ```

## Code Overview
1. **Loading the YOLO model**: Reads the model weights, configuration, and class names.
2. **Frame processing**: Converts each frame into a blob and passes it through the YOLO network.
3. **Detection filtering**: Extracts class IDs and confidence scores, applies thresholding and NMS.
4. **Drawing boxes**: Labels detected objects with bounding boxes and confidence scores.
5. **Displaying results**: Shows the processed frames in a window.
6. **Exiting the program**: Press `Esc` to quit.

## Example Output
When running the script, you should see video frames with detected objects highlighted.

## Troubleshooting
- If `yolov3.weights` is missing, download it from the provided link.
- If OpenCV fails to open the video, ensure the file path is correct.
- If using a webcam, verify that it's properly connected and accessible.

## License
This project is for educational purposes and follows the YOLO open-source licensing.

---

Enjoy real-time object detection with YOLOv3! 🚀
