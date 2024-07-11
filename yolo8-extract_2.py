import cv2
import argparse
from ultralytics import YOLO
import json
import csv
import os
import torch
import time

parser = argparse.ArgumentParser(description="YOLO8 layer data extractor")
parser.add_argument('image_path', type=str, help="Input image path")
parser.add_argument('--layer', type=str, default='model.model.22.cv3.2.2.dfl', help="Layer name to data extract")
parser.add_argument('--list', action='store_true', help="Print names of all layers")
parser.add_argument('--output', type=str, default='model.model.22.cv3.2.2.json', help="Output file path")
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help="Device to run the model on (cpu or cuda)")

args = parser.parse_args()

# Check if the image file exists
if not os.path.isfile(args.image_path):
    raise FileNotFoundError(f"Image file not found: {args.image_path}")

img = cv2.imread(args.image_path)

# Check if the image was loaded correctly
if img is None:
    raise ValueError(f"Failed to load image: {args.image_path}")

# Print original image dimensions for debugging
print(f"Original image shape: {img.shape}")

# Check if image dimensions are valid
if len(img.shape) != 3 or img.shape[2] != 3:
    raise ValueError(f"Invalid image dimensions: {img.shape}")

device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
print(f"Using device: {device}")

# Explicitly print GPU information if CUDA is selected
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

model = YOLO("yolov8n.pt").to(device)

def hook_fn(module, input, output):
    np_list = output.cpu().numpy().tolist()  # Ensure the tensor is moved to CPU before converting

    # Print data type and shape of the output tensor
    print(f"Data type: {output.dtype}")
    print(f"Shape: {output.shape}")

    # Save as JSON
    json_data = json.dumps(np_list)
    with open(args.output, 'w') as json_file:
        json_file.write(json_data)

    # Save as CSV
    csv_output_path = os.path.splitext(args.output)[0] + '.csv'
    with open(csv_output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(np_list)

for name, layer in model.named_modules():
    if args.list:
        print(name)
    if args.layer == name:
        layer.register_forward_hook(hook_fn)

# Resize the image to be divisible by the model's stride (e.g., 640x640)
resize_dim = (640, 640)
try:
    img_resized = cv2.resize(img, resize_dim, interpolation=cv2.INTER_LINEAR)
    # Print resized image dimensions for debugging
    print(f"Resized image shape: {img_resized.shape}")
except Exception as e:
    print(f"Error resizing image: {e}")
    raise ValueError(f"Failed to resize image: {args.image_path}")

# Convert the resized image to a PyTorch tensor and normalize it
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)

start_time = time.time()
model(img_tensor)  # Ensure the image tensor is moved to the appropriate device
end_time = time.time()

print(f"Time spent processing the image: {end_time - start_time:.4f} seconds")
