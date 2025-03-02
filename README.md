# Brain Stroke Detection System

A full-stack web application for brain stroke detection and segmentation from MRI scans using FastAPI, PyTorch, and YOLO.

## Features

- Upload and analyze MRI brain scan images
- Two-model approach:
  - Classification model (CNN) to determine if a stroke is present
  - Segmentation model (YOLO) to identify and highlight stroke regions
- Interactive web interface with real-time image preview
- Visualization of classification probabilities
- Display of segmentation results

## Setup and Installation

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended for faster inference)

### Option 1: Using Docker

1. Clone the repository
   ```
   git clone https://github.com/Sirfowahid/BrainStrokeDetection.git
   cd BrainStrokeDetection
   ```

2. Build the Docker image
   ```
   docker build -t brain-stroke-detection .
   ```

3. Run the Docker container
   ```
   docker run -p 8000:8000 -v $(pwd)/models:/app/models brain-stroke-detection
   ```

4. Access the application at http://localhost:8000

### Option 2: Manual Setup

1. Clone the repository
   ```
   git clone https://github.com/Sirfowahid/brain-stroke-detection.git
   cd brain-stroke-detection
   ```

2. Create a virtual environment and activate it
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Create necessary directories
   ```
   mkdir -p models static/uploads
   ```

5. Download model weights
   - Place the CNN model at `models/best_model.pth`
   - Place the YOLO model at `models/best.pt`

6. Run the application
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. Access the application at http://localhost:8000

## Project Structure

```
brain-stroke-detection/
├── main.py                 # FastAPI application 
├── model.py                # Model definitions
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── models/                 # Model weights
│   ├── best_model.pth      # CNN model
│   └── best.pt             # YOLO model
├── static/                 # Static files
│   └── uploads/            # Uploaded and processed images
└── templates/              # HTML templates
    ├── index.html          # Upload page
    └── result.html         # Results page
```

## Model Information

### Classification Model (CNN)

The BioClassNet model is a custom CNN architecture for binary classification of MRI images:
- Input: 224x224 RGB images
- Output: Binary classification (Stroke/Non-Stroke)
- Architecture: 4 convolutional blocks with batch normalization and max pooling

### Segmentation Model (YOLO)

The YOLO v9/v10 model for object detection and segmentation:
- Trained on the Updated-Mri-based-brain-stroke dataset
- Detects and segments stroke regions in MRI images

## API Endpoints

- `GET /`: Main page for image upload
- `POST /analyze`: Processes uploaded image and returns analysis results
- `GET /api/health`: Health check endpoint

## Medical Disclaimer

This application is for research and educational purposes only. The results should not be used for medical diagnosis without consultation with healthcare professionals.

## License

[MIT License](LICENSE)
