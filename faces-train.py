import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
image_dir = os.path.join(BASE_DIR, "images")
print(image_dir)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

print("training...")
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			#print(label, path)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]

			pil_image = Image.open(path).convert("L") # grayscale
			size = (550, 550)
			final_image = pil_image.resize(size, Image.Resampling.LANCZOS)
			image_array = np.array(pil_image, "uint8")
			
			faces = face_cascade.detectMultiScale(image_array)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)


#print(y_labels)
#print(x_train)

filename = "pickles\\face-labels.pickle"
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open( "pickles\\face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))

filename = "recognizers\\face-trainner.yml"
os.makedirs(os.path.dirname(filename), exist_ok=True)
recognizer.save("recognizers\\face-trainner.yml")
print("training completed")