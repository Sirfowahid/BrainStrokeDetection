# main.py
import os
import io
import uvicorn
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
from model import BioClassNet, YOLOModel

app = FastAPI(title="Brain Stroke Detection API")

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
MODEL_PATH = "models/best_model.pth"
YOLO_MODEL_PATH = "models/best.pt"
CLASSES = ['Acute Stroke', 'Cerebral Hemorrhage', 'Fatal Stroke', 'Multiple Embolic Infarctions', 'Non Stroke']
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BioClassNet(num_classes=len(CLASSES)).to(device)

# Load model weights
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"CNN Model loaded successfully from {MODEL_PATH}")
else:
    print(f"Warning: Model file {MODEL_PATH} not found. Please train the model first.")

# Load YOLO model for segmentation
yolo_model = None
if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLOModel(YOLO_MODEL_PATH)
    print(f"YOLO Model loaded successfully from {YOLO_MODEL_PATH}")
else:
    print(f"Warning: YOLO model file {YOLO_MODEL_PATH} not found.")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...), model_type: str = Form("cnn")):
    # Save the uploaded file
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Open the image
    img = Image.open(file_location).convert('RGB')
    
    result = {}
    
    # CNN Classification
    if model_type == "cnn" or model_type == "both":
        # Preprocess the image
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        # Format results
        result["cnn"] = {
            "prediction": CLASSES[predicted_class],
            "confidence": float(probabilities[predicted_class]) * 100,
            "probabilities": {cls: float(prob) * 100 for cls, prob in zip(CLASSES, probabilities)}
        }
    
    # YOLO Segmentation
    if (model_type == "yolo" or model_type == "both") and yolo_model is not None:
        yolo_result = yolo_model.predict(file_location)
        result_image_path = os.path.join(UPLOAD_FOLDER, f"result_{file.filename}")
        
        if yolo_result:
            yolo_result[0].save(filename=result_image_path)
            result["yolo"] = {
                "detection": True,
                "result_image": f"/static/uploads/result_{file.filename}"
            }
        else:
            result["yolo"] = {
                "detection": False,
                "message": "No stroke regions detected in the image."
            }
    
    return templates.TemplateResponse("result.html", {
        "request": request, 
        "result": result,
        "original_image": f"/static/uploads/{file.filename}"
    })

@app.get("/api/health")
async def health_check():
    return {"status": "online", "models": {"cnn": os.path.exists(MODEL_PATH), "yolo": os.path.exists(YOLO_MODEL_PATH)}}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)