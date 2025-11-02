# FILE: interpret_cam.py
# (This is the complete, final, and corrected file)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
# import cv2  <- REMOVED
import numpy as np
import os
import timm
import streamlit as st
import matplotlib.pyplot as plt # <-- ADDED

# --- 1. Define Our Constants ---
CLASS_NAMES = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
NUM_CLASSES = len(CLASS_NAMES)

# --- 2. Define Our Model "Blueprints" and "Weights" ---
MODEL_CONFIG = {
    "ResNet50": {
        "model_file": "models/resnet50_best_p15.pth", # 91.25% Test Acc
        "builder": "resnet50",
        "target_layer": lambda m: [m.layer4[-1]],
        "reshape_transform": None
    },
    "AlexNet": {
        "model_file": "models/alexnet_best_p30.pth", # 90.85% Test Acc
        "builder": "alexnet",
        "target_layer": lambda m: [m.features[-1]],
        "reshape_transform": None
    },
    "EfficientNet-B4": {
        "model_file": "models/efficient_b4_best.pt", # 87.90% Test Acc
        "builder": "efficientnet",
        "target_layer": lambda m: [m.conv_head],
        "reshape_transform": None
    },
    "ViT-Base-Patch16": {
        "model_file": "models/vit_base_patch16_best.pt", # 78.80% Test Acc
        "builder": "vit",
        "target_layer": lambda m: [m.blocks[-1].norm1],
        "reshape_transform": lambda t: t.reshape(t.shape[0], int(t.shape[1]**0.5), int(t.shape[1]**0.5), t.shape[2]).permute(0, 3, 1, 2)
    }
}
MODEL_LIST = list(MODEL_CONFIG.keys())
DEFAULT_MODEL = MODEL_LIST[0] # "ResNet50 (Champion)"

# --- 3. The "Engine" (Caching & Loading) ---
# (build_model_skeleton, load_model_weights, get_transform, 
# and process_image_for_cam are UNCHANGED. You can keep your existing ones
# or paste this block.)

@st.cache_resource
def build_model_skeleton(model_name):
    print(f"Cache miss: Building {model_name} skeleton...")
    config = MODEL_CONFIG[model_name]
    if config["builder"] == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif config["builder"] == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    elif config["builder"] == "vit":
        model = timm.create_model("vit_base_patch16_siglip_224", pretrained=False, num_classes=NUM_CLASSES)
    elif config["builder"] == "efficientnet":
        model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=NUM_CLASSES)
    else: raise ValueError(f"Unknown builder: {config['builder']}")
    model.eval()
    return model

@st.cache_resource
def load_model_weights(model_name):
    print(f"Cache miss: Loading weights for {model_name}...")
    model = build_model_skeleton(model_name)
    config = MODEL_CONFIG[model_name]
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(config["model_file"], map_location=device, weights_only=True))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def process_image_for_cam(pil_image):
    transform_model = get_transform()
    input_tensor = transform_model(pil_image).unsqueeze(0) 
    transform_viz = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    rgb_img_tensor = transform_viz(pil_image)
    rgb_img = np.float32(rgb_img_tensor.permute(1, 2, 0)) 
    return input_tensor, rgb_img

# --- 4. The Main Public Function (Our App Will Call This) ---
# --- THIS FUNCTION IS NOW 100% CORRECTED ---

def generate_cam_visualization(model_name, pil_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_weights(model_name)
    if model is None: return None, "Error: Model not loaded."
    model.to(device)
    config = MODEL_CONFIG[model_name]
    target_layer_func = config["target_layer"]
    reshape_transform = config["reshape_transform"]
    input_tensor, vis_image = process_image_for_cam(pil_image)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
    pred_label = CLASS_NAMES[pred_idx]
    confidence = probabilities[0, pred_idx].item()
    
    # --- Import statements for Grad-CAM ---
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image 
    
    target_layers = target_layer_func(model)
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # 'visualization' is an RGB float (0-1), which is what matplotlib wants
    visualization = show_cam_on_image(vis_image, grayscale_cam, use_rgb=True)
    
    # --- THIS IS THE NEW FIX: Use Matplotlib, not CV2 ---
    
    # 1. Create a new figure and axis
    fig, ax = plt.subplots()
    
    # 2. Display the RGB visualization
    ax.imshow(visualization)
    
    # 3. Add the title (same as your teammate's style)
    title = f"Prediction: {pred_label} ({confidence*100:.2f}%)"
    ax.set_title(title, fontsize=12, color='black', pad=10) # Added padding
    
    # 4. Turn off the axes (cleaner look)
    ax.axis('off')
    
    # 5. Save the figure to a bytes buffer
    # This is a "virtual file" in memory that Streamlit can read
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight') # Use bbox_inches to prevent cutoff
    buf.seek(0)
    
    # 6. Close the figure to free memory
    plt.close(fig)
    
    # 7. Return the image (from the buffer) and the label
    return buf, pred_label