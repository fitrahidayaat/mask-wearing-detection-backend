from ultralytics import YOLO
import cv2

def yolo_inference(model_path, image_path, img_size=416, conf_threshold=0.5):
    # Load the pretrained YOLO model
    model = YOLO(model_path)
    
    # Run inference on the image
    results = model.predict(image_path, save=True, imgsz=img_size, conf=conf_threshold)
    
    # Get the image with detections drawn on it
    img = results[0].plot()  # This plots the detections on the image
    
    return img
