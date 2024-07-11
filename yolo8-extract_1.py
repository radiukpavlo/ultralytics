import cv2
import argparse
from ultralytics import YOLO
import json

parser = argparse.ArgumentParser(description="YOLO8 layer data extractor")
parser.add_argument('image_path', type=str, help="Input image path")
parser.add_argument('--layer', type=str, help="Layer name to data extract")
parser.add_argument('--list', default='True', help="Print names of all layers")
parser.add_argument('--output', type=str, default='output.json', help="Output file path")

args = parser.parse_args()
img = cv2.imread(args.image_path)

model = YOLO("yolov8n.pt")


def hook_fn(module, input, output):
    np_list = output.numpy().tolist()
    json_data = json.dumps(np_list)    

    with open(args.output, 'w') as file:
        file.write(json_data)


for name, layer in model.named_modules():        
    if args.list:
        print(name)
    if args.layer == name:
        layer.register_forward_hook(hook_fn)

model(img) 
        