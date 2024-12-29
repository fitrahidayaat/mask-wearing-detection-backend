from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

# Initialize YOLO model once
MODEL_PATH = "best.torchscript"  # Replace with your actual model path

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your Next.js app's URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_model():
    # Load the model at startup to avoid reloading on each request
    print("Loading the YOLO model...")
    global model  # Declare model as global to use it across functions
    model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Read image file into memory
        image_bytes = await image.read()
        img = Image.open(BytesIO(image_bytes))

        # Save the image to a temporary file or process it directly
        img.save("temp_image.jpg")  # You can skip saving and pass it directly to yolo_inference if preferred

        # Run inference on the image
        results = model.predict("temp_image.jpg", save=True, imgsz=416, conf=0.5)

        # Get the image with detections drawn on it
        img = results[0].plot()  # This plots the detections on the image

        # Convert the result (OpenCV format) to a base64 string
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return JSONResponse(
            content={"prediction": img_base64}
        )
    except Exception as e:
        print(str(e))
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
