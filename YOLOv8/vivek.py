from ultralytics import YOLO
import cv2


model = YOLO("best.pt")

model.predict(source=1, show=True)