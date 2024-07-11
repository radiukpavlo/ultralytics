"""
test_1.py
"""
import cv2
import torch
from ultralytics import YOLO


# Function to perform object detection on an image
def detect_objects(image_path, device='cpu'):
    # Load the YOLOv8 model and move it to the specified device
    model = YOLO('yolov8n.pt').to(device)

    # Load the image
    img = cv2.imread(image_path)

    # Perform object detection
    results = model(img)

    # Get bounding boxes and labels from the results
    for result in results:
        boxes = result.boxes

        print("\nDetection results:")
        # print(boxes)

        for box in boxes:
            # Extract bounding box coordinates
            coords = box.xyxy[0].to('cpu').numpy().astype(int)
            xmin, ymin, xmax, ymax = coords
            # Extract confidence score
            confidence = box.conf.item()
            # Extract class ID
            class_id = box.cls.item()
            # Get class name
            name = result.names[class_id]
            # Create label with class name and confidence
            label = f"{name} {confidence:.2f}"
            # Define bounding box color (green)
            color = (0, 255, 0)

            # Print values of boxes, confidence, and class_id
            # print(f"\nCoordinates: {coords}")
            # print(f"Box coordinates: (xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax})")
            # print(f"Confidence score: {confidence}")
            # print(f"Class ID: {class_id}")

            # Draw bounding box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

            # Put label text above the bounding box
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x = xmin
            label_y = ymin - label_size[1] if ymin - label_size[1] > 10 else ymin + label_size[1]
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save or display the image with detections
    output_path = 'output_bus.jpg'
    cv2.imwrite(output_path, img)
    print(f"\nDetection completed. Results saved to {output_path}")


# Path to the input image
image_path = 'bus.jpg'  # Replace with your image path

# Choose device: 'cuda' or 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Detect objects on the image
detect_objects(image_path, device)
