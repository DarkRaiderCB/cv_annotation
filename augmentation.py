import os
import random
import cv2
import numpy as np


def augment_image(image):
    # (0: vertical, 1: horizontal, -1: both)
    if random.choice([True, False]):
        flip_code = random.choice([-1, 0, 1])
        image = cv2.flip(image, flip_code)

    # blur
    if random.choice([True, False]):
        ksize = random.choice([3, 5, 7, 9, 11])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # brightness adjustment
    if random.choice([True, False]):
        value = random.randint(-100, 100)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v = np.clip(v.astype(int) + value, 0, 255).astype(np.uint8)

        hsv = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # color adjustment
    if random.choice([True, False]):
        factor = random.uniform(0.4, 1.6)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)

    # grayscale conversion
    if random.choice([True, False]):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def process_images(input_folder, output_folder, num_images):
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(
        input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    selected_files = random.sample(
        image_files, min(num_images, len(image_files)))

    for file_name in selected_files:
        img_path = os.path.join(input_folder, file_name)
        image = cv2.imread(img_path)

        augmented_image = augment_image(image)

        output_path = os.path.join(output_folder, f"aug_{file_name}")
        cv2.imwrite(output_path, augmented_image)


input_folder = "images"
output_folder = "augmented"
num_images_to_process = 25

process_images(input_folder, output_folder, num_images_to_process)

print("Image augmentation complete!")
