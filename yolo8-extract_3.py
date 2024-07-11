import importlib

import cv2
import argparse
import ultralytics
from ultralytics import YOLO
import json
import csv
import os
import torch
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO8 layer data extractor")
    parser.add_argument('--image_path', type=str, default='bus.jpg', help="Input image path")
    parser.add_argument('--layer', type=str, default='model.model.22.dfl', help="Layer name to data extract")
    parser.add_argument('--list', action='store_true', help="Print names of all layers")
    parser.add_argument('--output', type=str, default='model.model.22.dfl.json', help="Output file path")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to run the model on (cpu or cuda)")
    return parser.parse_args()


def check_image_file(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"\nImage file not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"\nFailed to load image: {image_path}")
    print(f"\nOriginal image shape: {img.shape}")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"\nInvalid image dimensions: {img.shape}")
    return img


def set_device(device_choice):
    device = torch.device(device_choice if torch.cuda.is_available() and device_choice == 'cuda' else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    return device


def create_model(device):
    model = YOLO("yolov8n.pt").to(device)
    return model


def hook_fn(module, input, output, hook):
    np_list = output.cpu().numpy().tolist()  # Ensure the tensor is moved to CPU before converting
    # print(f"Layer name: {name}")
    print("")
    print(f"Data type: {output.dtype}")
    print(f"Shape: {output.shape}")
    print("")

    # Save as JSON
    json_data = json.dumps(np_list)
    with open(args.output, 'w') as json_file:
        json_file.write(json_data)

    # Save as CSV
    # csv_output_path = os.path.splitext(args.output)[0] + '.csv'
    # with open(csv_output_path, 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerows(np_list)

    hook.remove()


def register_hooks(model, layer_name, list_layers):
    for name, layer in model.named_modules():
        if args.list:
            print(name)
        if args.layer == name:
            hook = layer.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, hook))
            print(f"\nHook registered for layer: {name}")
            break


# def register_hooks(model, layer_name, list_layers):
#     for name, layer in model.named_modules():
#         hook = layer.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, hook))
#         print(f"\nHook registered for layer: {name}")


def resize_image(img, resize_dim=(640, 640)):
    try:
        img_resized = cv2.resize(img, resize_dim, interpolation=cv2.INTER_LINEAR)
        print(f"Resized image shape: {img_resized.shape}")
    except Exception as e:
        print(f"\nError resizing image: {e}")
        raise ValueError(f"\nFailed to resize image: {args.image_path}")
    return img_resized


def convert_image_to_tensor(img_resized, device):
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
    return img_tensor


def main():
    global args
    args = parse_args()

    device = set_device(args.device)
    model = create_model(device)

    # for name, layer in model.named_modules():
    #     if args.list:
    #         print(name)
    #     if args.layer == name:
    #         layer.register_forward_hook(hook_fn)

    img = check_image_file(args.image_path)
    img_resized = resize_image(img)
    img_tensor = convert_image_to_tensor(img_resized, device)

    register_hooks(model, args.layer, args.list)

    # Display the model's architecture
    # print(model)

    start_time = time.time()
    results = model(img_tensor)

    # print(f"\nThe final results on the last layer: ")
    # print(results)

    end_time = time.time()

    print(f"\nTime spent processing the image: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()

#%%
