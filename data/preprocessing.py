import os
import cv2
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Directory paths
data_dir = r"C:\Users\User\OneDrive\Desktop\Computational Intelligence\Assignment 2\malaysian_food"
processed_dir = r"C:\Users\User\OneDrive\Desktop\Computational Intelligence\Assignment 2\malaysian_food\processed_images"

# Classes to include
class_names = ['nasi_lemak', 'roti_canai', 'satay']

def create_folder_structure():
    """Creates the necessary folder structure for train, test, and validation datasets."""
    splits = ['train', 'validate', 'test']
    folders = [
        os.path.join(processed_dir, split, class_name)
        for split in splits
        for class_name in class_names
    ]
    # Create the directories if they don't exist
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def augment_image(image):
    """Applies random augmentations to an image."""
    # Flip horizontally
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    # Rotate randomly
    angle = random.uniform(-15, 15)  # Random rotation between -15 and 15 degrees
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (w, h))
    # Random brightness adjustment
    factor = random.uniform(0.8, 1.2)
    image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image

def preprocess_data():
    """Preprocesses the dataset and saves images into the proper directories."""
    images = []
    labels = []

    # Iterate over classes and their images
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = cv2.imread(image_path)

                # Check if the image is valid
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                # Resize and normalize the image
                image = cv2.resize(image, (224, 224))
                images.append(image)
                labels.append(label)

                # Create augmented data (add 100 additional images per class)
                if len(images) % 100 == 0:
                    for _ in range(100 // len(class_names)):
                        augmented_image = augment_image(image)
                        images.append(augmented_image)
                        labels.append(label)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save processed images to their respective folders
    save_images_to_folders(X_train, y_train, 'train')
    save_images_to_folders(X_val, y_val, 'validate')
    save_images_to_folders(X_test, y_test, 'test')

    print(f"Data preprocessing completed. Images saved in {processed_dir}.")
    return (X_train, y_train), (X_val, y_val)

def save_images_to_folders(images, labels, split):
    """Saves images into the respective folders based on split (train/test/validate)."""
    for i, (image, label) in enumerate(zip(images, labels)):
        class_name = class_names[label]
        class_folder = os.path.join(processed_dir, split, class_name)
        
        # Ensure the folder exists
        os.makedirs(class_folder, exist_ok=True)

        # Save the image
        image_name = f"{split}_{class_name}_{i}.jpg"
        image_path = os.path.join(class_folder, image_name)
        cv2.imwrite(image_path, image)

def check_class_distribution(base_path):
    """
    Checks the class distribution in the train, validate, and test datasets.

    Args:
        base_path (str): The base directory containing the train, validate, and test splits.
    """
    splits = ['train', 'validate', 'test']

    for split in splits:
        split_path = os.path.join(base_path, split)
        labels = []

        # Traverse through the dataset split folder
        for root, dirs, files in os.walk(split_path):
            for file in files:
                # Only count image files (can modify to check specific extensions if needed)
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    label = os.path.basename(root)
                    labels.append(label)

        # Count occurrences of each class
        distribution = Counter(labels)
        print(f"Class distribution in {split} set:")
        for class_name, count in distribution.items():
            print(f"  {class_name}: {count}")
        print()  # Add a blank line for readability

def visualize_random_training_images(base_path, num_images=20):
    """
    Visualizes random images from the training set with their class names.

    Args:
        base_path (str): The base directory containing the processed dataset.
        num_images (int): The number of random images to display.
    """
    train_path = os.path.join(base_path, "train")
    image_paths = []
    labels = []

    # Collect all image paths and their corresponding labels
    for class_name in class_names:
        class_folder = os.path.join(train_path, class_name)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_folder, image_name))
                    labels.append(class_name)

    # Randomly sample the images
    sampled_indices = random.sample(range(len(image_paths)), num_images)
    sampled_images = [cv2.imread(image_paths[i]) for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]

    # Create a figure to display the images
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for ax, image, label in zip(axes, sampled_images, sampled_labels):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_folder_structure()  # Create the directories
    preprocess_data()  # Process images and save them into respective folders
    check_class_distribution(processed_dir)  # Check class distribution
    visualize_random_training_images(processed_dir)  # Visualize random training images
