# -------------------------------------------------
# Image Processing Script
# -------------------------------------------------
# Purpose:
# 1. Read raw RGB images
# 2. Split into individual color channels (R, G, B)
# 3. Apply Sobel edge detection on each channel
# 4. Normalize the Sobel magnitude
# 5. Merge back into a 3-channel image
# 6. Save the processed images for YOLO training
# -------------------------------------------------

import cv2
import os
import glob
import numpy as np

# ---------------- CONFIG ----------------

# Path to input images (raw RGB images)
#change to own directory 
input_dir = "./images"

# Path to output images (Sobel-processed 3-channel images)
#change to own directory 
output_dir = "./sobel_images"

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Supported image file extensions
exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

# ---------------------------------------

def sobel_magnitude(channel):
    """
    Apply Sobel edge detection on a single channel.

    Steps:
    1. Compute Sobel gradients in X and Y directions
    2. Calculate gradient magnitude
    3. Normalize result to [0, 255]
    4. Convert to uint8 for image storage

    Args:
        channel (numpy.ndarray): Single-channel image

    Returns:
        numpy.ndarray: Normalized Sobel magnitude image
    """

    # Sobel gradient in X direction
    sobel_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)

    # Sobel gradient in Y direction
    sobel_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)

    # Compute gradient magnitude
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Normalize magnitude values to 0–255
    magnitude = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    )

    return magnitude.astype(np.uint8)

# ---------------------------------------

# Collect all image paths
image_paths = []
for ext in exts:
    image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

print(f"Found {len(image_paths)} images")

# Process each image
for img_path in image_paths:
    filename = os.path.basename(img_path)

    # Read image (OpenCV reads images in BGR format)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Failed to read {filename}")
        continue

    # Convert image from BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Split RGB image into individual channels
    R, G, B = cv2.split(img_rgb)

    # Apply Sobel magnitude on each color channel
    R_edge = sobel_magnitude(R)
    G_edge = sobel_magnitude(G)
    B_edge = sobel_magnitude(B)

    # Merge Sobel-processed channels back into a 3-channel image
    sobel_rgb = cv2.merge([R_edge, G_edge, B_edge])

    # Convert RGB back to BGR for saving
    sobel_bgr = cv2.cvtColor(sobel_rgb, cv2.COLOR_RGB2BGR)

    # Save processed image
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, sobel_bgr)

print("✅ RGB Sobel preprocessing completed.")
