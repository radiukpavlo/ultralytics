import cv2
import argparse
import onnxruntime as ort
import json
import os
import numpy as np
import time

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX model inference")
    parser.add_argument('--image_path', type=str, default='bus.jpg', help="Input image path")
    parser.add_argument('--output', type=str, default='output.json', help="Output file path")
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
    device = 'cuda' if (device_choice == 'cuda' and ort.get_device() == 'GPU') else 'cpu'
    print(f"\nUsing device: {device}")
    return device

def create_model(device):
    providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession('yolov8n.onnx', providers=providers)
    return session

def resize_image(img, resize_dim=(640, 640)):
    try:
        img_resized = cv2.resize(img, resize_dim, interpolation=cv2.INTER_LINEAR)
        print(f"Resized image shape: {img_resized.shape}")
    except Exception as e:
        print(f"\nError resizing image: {e}")
        raise ValueError(f"\nFailed to resize image: {args.image_path}")
    return img_resized

def convert_image_to_tensor(img_resized, device):
    img_tensor = np.transpose(img_resized, (2, 0, 1)).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return img_tensor

def decode_predictions(predictions, confidence_threshold=0.5):
    boxes = predictions[0]  # Assuming the first output is the bounding boxes
    scores = predictions[1]  # Assuming the second output is the scores
    classes = predictions[2]  # Assuming the third output is the class probabilities

    detected_objects = []
    for box, score, cls in zip(boxes, scores, classes):
        if score > confidence_threshold:
            detected_objects.append({
                "box": box,
                "score": score,
                "class": cls
            })
    return detected_objects

def main():
    global args
    args = parse_args()

    device = set_device(args.device)
    session = create_model(device)

    img = check_image_file(args.image_path)
    img_resized = resize_image(img)
    img_tensor = convert_image_to_tensor(img_resized, device)

    input_name = session.get_inputs()[0].name
    start_time = time.time()
    results = session.run(None, {input_name: img_tensor})  # Perform inference

    print(f"\nThe final results on the last layer: ")
    for result in results:
        print(f"Shape: {np.array(result).shape}")

    end_time = time.time()
    print(f"\nTime spent processing the image: {end_time - start_time:.4f} seconds")

    # Decode predictions
    decoded_predictions = decode_predictions(results)
    print("\nDecoded Predictions:")
    for pred in decoded_predictions:
        print(f"Box: {pred['box']}, Score: {pred['score']}, Class: {pred['class']}")

    # Save results to JSON
    with open(args.output, 'w') as json_file:
        json.dump([r.tolist() for r in results], json_file)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
