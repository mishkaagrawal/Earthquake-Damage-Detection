from flask import Flask, render_template, request, redirect, url_for, jsonify
from ultralytics import YOLO
import cvzone
import cv2
from datetime import datetime
import pickle
import random
import os
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'bmp', 'mp4'}

# Load earthquake damage detection model
model = YOLO('models/model.pt')
classnames = ['damage']  # Define class names for earthquake damage
detections = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_image(path):
    """Checks if the path points to an image file."""
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    return path.lower().endswith(extensions)

def process_file(path):
    global detections
    detections = []

    if is_image(path):
        frame = cv2.imread(path)
        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)
        process_detections(result, frame)
        cv2.imshow("Processing Image", frame)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            result = model(frame, stream=True)
            process_detections(result, frame)
            cv2.imshow("Processing Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # Save detections to a file
    with open('detections.pkl', 'wb') as f:
        pickle.dump(detections, f)
    print(f"Detections: {detections}")

def process_detections(result, frame):
    global detections
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            confidence = round(confidence * 100, 4)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)
                detection_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                criticalness = max(10, confidence - 53)
                if criticalness == 10:
                    criticalness = round(random.uniform(10, 11), 3)
                else:
                    criticalness = round(criticalness, 4)

                # Encode the frame to save as a snapshot
                _, buffer = cv2.imencode('.jpg', frame)
                snapshot = buffer.tobytes()

                # Append detection details
                detections.append((classnames[Class], detection_time, criticalness, confidence, snapshot))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        process_file(filepath)
        return redirect(url_for('index'))
    return redirect(request.url)

@app.route('/detections', methods=['GET'])
def get_detections():
    with open('detections.pkl', 'rb') as f:
        detections = pickle.load(f)
    
    # Prepare detections for JSON serialization
    json_detections = []
    for detection in detections:
        class_type, detection_time, criticalness, confidence, snapshot = detection
        # Encode snapshot to base64
        snapshot_base64 = base64.b64encode(snapshot).decode('utf-8')
        json_detections.append({
            'Type': class_type,
            'Time': detection_time,
            'Criticalness': criticalness,
            'Accuracy': confidence,
            'Snapshot': snapshot_base64
        })
    
    return jsonify(json_detections)

if __name__ == '__main__':
    app.run(debug=True)
