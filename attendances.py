import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Face detector
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load labels (ID → Name)
labels = {}
with open("labels.txt", "r") as f:
    for line in f:
        id, name = line.strip().split(",")
        labels[int(id)] = name

# Webcam
cam = cv2.VideoCapture(0)

marked = set()

def mark_attendance(name):
    file = "attendance.csv"

    # create file if not exists
    if not os.path.exists(file):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(file, index=False)

    df = pd.read_csv(file)

    # avoid duplicate entry
    if name in df["Name"].values:
        return

    now = datetime.now()

    new_row = {
        "Name": name,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S")
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(file, index=False)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        print("Prediction ID:", id, "Confidence:", confidence) 

        if confidence < 60:
            name = labels.get(id, "Unknown")

            mark_attendance(name)

            cv2.putText(img, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Attendance System", img)

    if cv2.waitKey(1) == 27:  # press ESC to exit
        break

cam.release()
cv2.destroyAllWindows()