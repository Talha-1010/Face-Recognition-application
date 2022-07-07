import string
import random
from turtle import bgcolor, color
import numpy as np
import cv2
import pickle

# UI libraries
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import shutil


# face train libraries
from PIL import ImageTk, Image
import os
from os.path import abspath


root = tk.Tk()
root.title("Face Recognizer")


def open_file():
    filename = filedialog.askopenfilename(initialdir="\\", title="Select An Image", filetypes=(
        ("jpeg files", "*.jpg"), ("gif files", "*.gif*"), ("png files", "*.png")))
    original = filename
    target = "C:\\Users\\DELL\\Documents\\Computer Vision\\Face Recognition application\\images\\"+result
    shutil.copyfile(original, target)


def face_detection():

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades+'haarcascade_smile.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read("recognizers\\face-trainner.yml")

    labels = {"person_name": 1}
    with open("pickles\\face-labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # recognize?

            # Check if confidence is less them 100 ==> "0" is perfect match


            # if (confidence < 100):
            #     id = names[id]
            #     confidence = "  {0}%".format(round(100 - confidence))
            # else:
            #     id = "unknown"
            #     confidence = "  {0}%".format(round(100 - confidence))


            id_, conf = recognizer.predict(roi_gray)
            if  conf <= 100:
                confidence = "  {0}%".format(round(100 - conf))
                # print(5: #id_)
                # print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x+5, y-5), font, 1,
                            color, stroke, cv2.LINE_AA)
                cv2.putText(frame, confidence, (x+5, y+h-5), font, 1,
                            color, stroke, cv2.LINE_AA)
            elif conf>100:

                confidence = "  {0}%".format(round(100 - conf))
                color = (255, 255, 255)
                name = "unknown"
                stroke = 2
                cv2.putText(frame, name, (x + 5, y - 5), font, 1,
                            color, stroke, cv2.LINE_AA)
                cv2.putText(frame, confidence, (x, y+h-5), font, 1,
                            color, stroke, cv2.LINE_AA)


            img_item = "7.png"
            cv2.imwrite(img_item, roi_color)

            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x,
                          end_cord_y), color, stroke)
            # subitems = smile_cascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in subitems:
            #	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # Display the resulting frame

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def getFolderName():
    global result  # to return calculation
    result = str(E1.get())
    return result


# result = 0


def captureImage():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            print("press q, to close")
            break
        elif cv2.waitKey(32) & 0xFF == ord(' '):
            # SPACE pressed
            while img_counter<30:
                folderName = getFolderName()
                dir = os.path.join(
                    "C:\\Users\\DELL\\Documents\\Computer Vision\\Face Recognition application\\", "images", folderName)
                if not os.path.exists(dir):
                    os.mkdir(dir)
                img_name = "{}.jpg".format(img_counter)
                img_location = os.path.join(dir,img_name)
                cv2.imwrite(img_location, frame)
                print("{} written!".format(img_name))
                img_counter += 1

    cam.release()

    cv2.destroyAllWindows()



def new_Face():
    filename = filedialog.askopenfilename(initialdir="\\", title="Select An Image", filetypes=(
        ("jpeg files", "*.jpg"), ("gif files", "*.gif*"), ("png files", "*.png")))
    print("filename == ", filename)

    head, tail = os.path.split(filename)

    original = filename
    S = 10
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k=S))
    folderName = getFolderName()
    print(os.curdir)
    dir = os.path.join(
       "C:\\Users\\DELL\\Documents\\Computer Vision\\Face Recognition application\\", "images", folderName)
    if not os.path.exists(dir):
        os.mkdir(dir)
        target = dir+"\\"+tail
    else:
        target = dir+"\\"+tail
    shutil.copyfile(original, target)


def face_train():
    filename = "faces-train.py"
    with open(filename, "rb") as source_file:
        code = compile(source_file.read(), filename, "exec")
    exec(code)





frame1 = LabelFrame(root, text="", padx=20, pady=20, background="#006994")
frame1.pack(fill=BOTH, expand=True, padx=10, pady=10)

# Create an object of tkinter ImageTk
img = ImageTk.PhotoImage(Image.open("face.jpg"))
# Create a Label Widget to display the text or Image
label = Label(frame1, image=img)
label.grid(row=1, column=20, padx=200, pady=10)


# creating frame
frame2 = LabelFrame(root, text="Options", padx=20, pady=20)
frame2.pack(fill=BOTH, expand=True, padx=10, pady=10)


E1 = Entry(frame2, bd=10, background="#006994")
E1.insert(0, 'Enter Face Name ')
# E1.pack(side = RIGHT)
E1.grid(row=1, column=0)


my_button = Button(frame2, text="Add New Face",
                   command=new_Face, background="#006994")
my_button.grid(row=1, column=5, padx=10, pady=10)

my_button = Button(frame2, text="Train the system",
                   command=face_train, background="#006994")
my_button.grid(row=1, column=6, padx=10, pady=10)


my_button = Button(frame2, text="Start Recognizer",
                   command=face_detection, background="#006994")
my_button.grid(row=1, column=7, padx=10, pady=10)


my_button = Button(frame2, text="quit",
                   command=root.destroy, background="#006994")
my_button.grid(row=1, column=9, padx=10, pady=10)

my_button = Button(frame2, text="capture images",
                   command=captureImage, background="#006994")
my_button.grid(row=1, column=8, padx=10, pady=10)


root.mainloop()
