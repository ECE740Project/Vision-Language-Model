## Dev: Abenanth ##
## UoA ID: 1836448 ##
## Email: kalyanaa@ualberta.ca ##

import cv2
import numpy as np
import os
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from bbox import BBox2D

# Directory paths
input_folder = "C:/Users/kalya/Desktop/Desktop/ML/lama/lama/coco/train2017"
# input_folder = "C:/Users/kalya/Desktop/Desktop/ML/lama/lama/test_aug"
# output_folder = "C:/Users/kalya/Desktop/Desktop/ML/lama/lama/output_all_images"
output_folder = "C:/Users/kalya/Desktop/Desktop/ML/lama/lama/output_all_images"
augmented_folder = os.path.join(output_folder, "augmented_images")
mask_folder = os.path.join(output_folder, "masks")
objects_folder = os.path.join(output_folder, "extracted_objects")

# Create directories for output, masks, and objects
os.makedirs(output_folder, exist_ok=True)
os.makedirs(augmented_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)
os.makedirs(objects_folder, exist_ok=True)

# List of all images in the input folder
image_files = [
    f for f in os.listdir(input_folder) if f.endswith((".jpg", ".jpeg", ".png"))
]

# Load YOLO
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n-seg.pt")
model.to(device)

# Gaussian noise function
# def add_gaussian_noise(image, mean=0, std=25):
#     noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
#     noisy_image = cv2.add(image, noise)
#     return noisy_image


if __name__ == "__main__":
    # Process each image in the folder
    ctr = 0
    for image_file in image_files:
        # Load the image
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        clone = image.copy()  # Keep a clone of the original image
        # Object detection with lower confidence threshold
        results = model(image, conf=0.5)
        detections = results[0].boxes
        img = plt.imread(str(image_file))
        height, width = img.shape[0] , img.shape[1]
        area_of_image = height * width
        bb_counter = 0
        # Loop through each detected object and get the bounding box coordinates
        for i, box in enumerate(detections):
            ctr += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            # Prepare mask for the detected objects (black background)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            print(f"for iterator = {i} and iamge = {image_file} x1 = {x1} y1 = {y1} x2 = {x2} y2 = {y2}")
            confidence = box.conf.cpu().numpy()
            bounding_box = BBox2D([x1, y1, x2, y2])
            area_of_bb = bounding_box.height * bounding_box.width
            percentage_of_area_covered_by_bb = (area_of_bb / area_of_image) * 100
            print("percentage_of_area_covered_by_bb = ",percentage_of_area_covered_by_bb)
            if percentage_of_area_covered_by_bb < 60:
                if ctr % 3 == 0:
                    obj_image = image[y1:y2, x1:x2]
                    # obj_image_with_noise = add_gaussian_noise(obj_image)
                    augmented_object_path = os.path.join(
                        objects_folder, f"object_{i}_{os.path.splitext(image_file)[0]}.png"
                    )
                    cv2.imwrite(augmented_object_path, obj_image)
                    # # Add Gaussian noise to the background image and save it
                    augmented_original_image = clone
                    # augmented_image_path = os.path.join(
                    #     augmented_folder, f"{os.path.splitext(image_file)[0]}_{i}.png"
                    # )
                    augmented_image_path = os.path.join(
                        augmented_folder, f"{os.path.splitext(image_file)[0]}_{i}.png"
                    )
                    cv2.imwrite(augmented_image_path, clone)
                    # Save the mask
                    mask[y1:y2, x1:x2] = 255
                    tmp = str(image_file)
                    img = tmp.split(".")[0]
                    num = tmp.split("e")[-1].split(".")[0]
                    mask_path = os.path.join(mask_folder, f"{img}_{i}_mask{num}_{i}.png")
                    cv2.imwrite(mask_path, mask)

                else:
                    obj_image = image[y1:y2, x1:x2]
                    obj_path = os.path.join(
                        objects_folder, f"object_{i}_{os.path.splitext(image_file)[0]}.png"
                    )
                    cv2.imwrite(obj_path, obj_image)
                    # No Gaussian noise to the background image and save it
                    # augmented_original_image = add_gaussian_noise(clone)
                    augmented_image_path = os.path.join(
                        augmented_folder, f"{os.path.splitext(image_file)[0]}_{i}.png"
                    )
                    cv2.imwrite(augmented_image_path, augmented_original_image)
                    # Save the mask
                    mask[y1:y2, x1:x2] = 255
                    tmp = str(image_file)
                    img = tmp.split(".")[0]
                    num = tmp.split("e")[-1].split(".")[0]
                    mask_path = os.path.join(mask_folder, f"{img}_{i}_mask{num}_{i}.png")
                    cv2.imwrite(mask_path, mask)
            else:
                continue

        print("Processing complete for all images.")










            




















