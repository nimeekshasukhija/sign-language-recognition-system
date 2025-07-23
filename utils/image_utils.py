import cv2
import numpy as np

def preprocess_image(image, size=(64, 64)):
    """Resize, normalize, and expand dims for prediction"""
    image = cv2.resize(image, size)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def get_roi(frame, x1=100, y1=100, x2=300, y2=300):
    """Extract ROI from frame"""
    roi = frame[y1:y2, x1:x2]
    return roi
