import os
import json
import csv
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def process_image(image):
    """
    Normalize and flatten the image.

    Args:
        image (torch.Tensor): The image tensor.

    Returns:
        np.ndarray: The flattened image.
    """
    # Normalize to [0, 1]
    image = image.cpu().numpy().transpose((1, 2, 0))
    image = image / 255.0
    # Flatten the image
    flattened_image = image.flatten()
    return flattened_image


def extract_image_id_from_filename(filename):
    """
    Extract the image ID from the filename by removing leading zeros.

    Args:
        filename (str): The image filename.

    Returns:
        int: The extracted image ID.
    """
    base = os.path.basename(filename)
    image_id_str = os.path.splitext(base)[0]  # Remove file extension
    image_id = int(image_id_str.lstrip('0'))  # Remove leading zeros and convert to int
    return image_id


def main(num_images, num_workers):
    """
    Main function to process the COCO dataset.

    Args:
        num_images (int): The number of images to process.
        num_workers (int): The number of worker processes to use for data loading.
    """
    # Define the root directory where COCO dataset is stored
    root_dir = r'D:\dataset\dataset_coco\coco2017'
    ann_file = os.path.join(root_dir, r'annotations\instances_train2017.json')
    img_dir = os.path.join(root_dir, 'train2017')

    # Print paths for debugging
    print(f"Root directory: {root_dir}")
    print(f"Annotation file: {ann_file}")
    print(f"Image directory: {img_dir}")

    # Check if a GPU is available and use it if possible
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f"Using device: {device}")

    # Define the transformation: resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load the COCO training dataset
    coco_train = datasets.CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transform=transform
    )

    # Limit the dataset to the specified number of images
    if num_images is not None:
        coco_train = Subset(coco_train, list(range(min(num_images, len(coco_train)))))

    # Create a DataLoader for batch processing
    train_loader = DataLoader(coco_train, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    # List to store all processed images
    all_images_data = []

    # Iterate through the DataLoader and process each image
    for i, (images, targets) in enumerate(train_loader):
        try:
            # Move images to the specified device
            images = images.to(device)

            # Debugging output to check data types
            print(f"Processing image {i + 1}/{len(train_loader)}")

            for j in range(len(images)):
                # Check if targets is not empty
                if not targets[j]:
                    print(f"Skipping image {i} due to empty targets")
                    continue

                # Process the image
                processed_image = process_image(images[j])

                # Extract the image ID from targets or filename
                if targets[j]:
                    image_id = targets[j][0]['image_id'].item()  # Convert tensor to int
                else:
                    image_id = extract_image_id_from_filename(coco_train.dataset.coco.loadImgs(i)[0]['file_name'])

                # Create a dictionary to store the image data
                image_data = {
                    'image_id': image_id,
                    'image_data': processed_image.tolist()  # Convert numpy array to list for JSON serialization
                }

                # Append the processed image data to the list
                all_images_data.append(image_data)

                if i % 100 == 0:
                    print(f'Processed {i} images')
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except IndexError as e:
            print(f"Skipping image {i} due to IndexError: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    print(f"\nNumber of images processed: {len(all_images_data)}")

    # Save all processed images to a single JSON file
    json_file = os.path.join(root_dir, 'coco_processed_images.json')
    with open(json_file, 'w') as f:
        json.dump(all_images_data, f)

    # Save all processed images to a CSV file
    csv_file = os.path.join(root_dir, 'coco_processed_images.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'image_data'])
        for image_data in all_images_data:
            writer.writerow([image_data['image_id'], json.dumps(image_data['image_data'])])

    print('All images processed and saved to JSON and CSV files.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a specified number of images from the COCO dataset.')
    parser.add_argument('--num_images', type=int, default=None, help='The number of images to process. If not specified, the whole dataset will be processed.')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of worker processes to use for data loading.')
    args = parser.parse_args()

    main(args.num_images, args.num_workers)
