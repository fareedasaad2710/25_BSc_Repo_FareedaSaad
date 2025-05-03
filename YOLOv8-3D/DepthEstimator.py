import torch
from transformers import AutoProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np

class DepthEstimator:
    def __init__(self):
        model_name = "LiheYoung/depth-anything-large-hf"  # New official HF version
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
        
    def predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt", size=(240, 320)).to(self.model.device)
        with torch.no_grad():
            depth = self.model(**inputs).predicted_depth
        return depth.squeeze(0).cpu().numpy()
