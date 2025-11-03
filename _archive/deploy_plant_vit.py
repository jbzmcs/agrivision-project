import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device("cuda")  # use CUDA only
model_path = "runs/train_9_vit/best_model.pt"  # Make sure this path is correct

# Load model
model = timm.create_model("vit_base_patch16_siglip_224", pretrained=False, num_classes=10)

# Fixed the safety warning
state_dict = torch.load(model_path, map_location=device, weights_only=True)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Labels
labels = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# =========================================
# TEST IMAGE PATHS
# =========================================
test_images = {
    "Tomato___Bacterial_spot": r"..\data\test\Tomato___Bacterial_spot\6855445b-ec04-42c0-bc12-ee8ff64420a0___GCREC_Bact.Sp 3268.jpg",
    "Tomato___Early_blight": r"..\data\test\Tomato___Early_blight\33cd99f1-1f8a-497b-b4fc-60ff78af4b8a___RS_Erly.B 8378.jpg",
    "Tomato___Late_blight": r"..\data\test\Tomato___Late_blight\8217ab61-6b5f-4874-8046-d801d40cff3d___RS_Late.B 6089.jpg",
    "Tomato___Leaf_Mold": r"..\data\test\Tomato___Leaf_Mold\6b56547b-6d59-40ef-852a-b3cf29f779b4___Crnl_L.Mold 8902.jpg",
    "Tomato___Septoria_leaf_spot": r"..\data\test\Tomato___Septoria_leaf_spot\4685caac-cd9f-45ff-b1c6-58bdc706d0c7___Matt.S_CG 6762.jpg",
    "Tomato___Spider_mites Two-spotted_spider_mite": r"..\data\test\Tomato___Spider_mites Two-spotted_spider_mite\737e3c49-a2d9-46f4-8c40-bcc5bc7343d2___Com.G_SpM_FL 1767.jpg",
    "Tomato___Target_Spot": r"..\data\test\Tomato___Target_Spot\44a9131e-6f2a-408d-ab7e-ccd0a935cdcc___Com.G_TgS_FL 9934.jpg",
    "Tomato___Tomato_mosaic_virus": r"..\data\test\Tomato___Tomato_mosaic_virus\5f6523e1-d5f2-44d9-a4a3-ae4a72a8a6cd___PSU_CG 2237.jpg",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": r"..\data\test\Tomato___Tomato_Yellow_Leaf_Curl_Virus\d7e6e51d-5643-4c15-8127-167f6b5161dc___UF.GRC_YLCV_Lab 02765.jpg",
    "Tomato___healthy": r"..\data\test\Tomato___healthy\085cbe78-1d5c-45eb-877f-f409526032d5___GH_HL Leaf 469.jpg",
}

# =========================================
# IMAGE TRANSFORM
# =========================================
def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

# =========================================
# ✅ MODIFIED FUNCTION TO FIX THE ERROR
# =========================================
def reshape_transform(tensor):
    # This model's output is (batch_size, 196, embedding_dim)
    # It does NOT have a [CLS] token to slice off.

    # ❌ DELETED: result = tensor[:, 1:, :]
    
    # The tensor is already just the patch tokens
    result = tensor 
    
    # Get the spatial dimensions (14x14)
    batch_size, num_tokens, channels = result.shape
    height = width = int(num_tokens**0.5) # 196**0.5 = 14
    
    # Reshape to (batch_size, height, width, channels)
    result = result.reshape(batch_size, height, width, channels)
    
    # Permute to (batch_size, channels, height, width)
    result = result.permute(0, 3, 1, 2)
    
    return result

# =========================================
# GRAD-CAM GENERATION
# =========================================
def apply_gradcam(image_path, model, device):
    
    image = Image.open(image_path).convert("RGB")
    
    # Resize image for visualization *before* transform
    vis_image = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Get model prediction
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    pred_label = labels[pred_idx]
    print(f"Predicted: {pred_label}")

    # Define the target layer for ViT
    target_layers = [model.blocks[-1].norm1]
    
    # Use 'GradCAM' and pass the 'reshape_transform'
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  reshape_transform=reshape_transform) # <-- This uses the fixed function

    # Define targets
    targets = [ClassifierOutputTarget(pred_idx)]

    # Generate the CAM
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    
    # Get the first (and only) CAM in the batch
    grayscale_cam = grayscale_cam[0, :]

    # Create the visualization
    visualization = show_cam_on_image(vis_image, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title(f"Prediction: {pred_label}")
    plt.axis("off")
    plt.show()

# =========================================
# LOOP THROUGH ALL TEST IMAGES
# =========================================
if __name__ == "__main__":
    for label, path in test_images.items():
        print(f"\nProcessing: {label}")
        try:
            apply_gradcam(path, model, device)
        except Exception as e:
            # ✅ Fixed the error message
            print(f"  !! ERROR processing {label}: {e}")