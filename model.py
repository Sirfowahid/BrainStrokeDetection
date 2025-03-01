# model.py
import torch
import torch.nn as nn
from ultralytics import YOLO

class BioClassNet(nn.Module):
    def __init__(self, num_classes=2):
        super(BioClassNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def predict(self, image_path, conf=0.25):
        """
        Run YOLO detection on an image
        
        Args:
            image_path: Path to the image file
            conf: Confidence threshold
            
        Returns:
            List of Results objects or None if no detections
        """
        results = self.model.predict(image_path, conf=conf)
        
        # Check if we have valid detections
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            return results
        return None