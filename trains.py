import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

label_map = {}
current_id = 0

dataset_path = "dataset"

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    print("Processing:", person)

    label_map[current_id] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(current_id)

    current_id += 1

print("Total faces:", len(faces))

if len(faces) == 0:
    print("ERROR: No images found in dataset")
    exit()

recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

with open("labels.txt", "w") as f:
    for id, name in label_map.items():
        f.write(f"{id},{name}\n")

print("Training Completed Successfully")