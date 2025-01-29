from ultralytics import YOLO
import cvzone
import cv2
import math
import os

# Load the model for earthquake damage detection
model = YOLO('model.pt')

# Define the class names
classnames = ['damage']  # Adjust based on your specific classes if needed

# File path for image or video
file_path = 'image2.jpg'  # Replace with your file name

def is_image(path):
    """Checks if the path points to an image file."""
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    return path.lower().endswith(extensions)

if is_image(file_path):
    # If the file is an image
    frame = cv2.imread(file_path)
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    # Process detection results
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    # Show the image and wait indefinitely
    cv2.imshow('Detection', frame)
    cv2.waitKey(0)  # Wait indefinitely for an image
    cv2.destroyAllWindows()

else:
    # If the file is a video
    cap = cv2.VideoCapture(file_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        # Process detection results
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                       scale=1.5, thickness=2)

        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
