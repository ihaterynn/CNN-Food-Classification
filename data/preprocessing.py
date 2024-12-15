import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Directory paths
data_dir = r"C:\Users\User\OneDrive\Desktop\Computational Intelligence\Assignment 2\Malaysian-11 Food Dataset"
processed_dir = r"C:\Users\User\OneDrive\Desktop\Computational Intelligence\Assignment 2\Malaysian-11 Food Dataset\processed_images"

def create_folder_structure():
    """Creates the necessary folder structure for train, test, and validation datasets."""
    # Define folder paths
    classes = ['nasi_lemak', 'roti_canai', 'satay']
    splits = ['train', 'validate', 'test']
    folders = [
        os.path.join(processed_dir, split, class_name)
        for split in splits
        for class_name in classes
    ]
    
    # Create the directories if they don't exist
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def preprocess_data():
    """Preprocesses the Malaysian Food-11 dataset and saves images into the proper directories."""
    images = []  
    labels = []  
    class_names = ['nasi_lemak', 'roti_canai', 'satay']

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
                image = image / 255.0  
                
                images.append(image)
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
    class_names = ['nasi_lemak', 'roti_canai', 'satay']
    for i, (image, label) in enumerate(zip(images, labels)):
        class_name = class_names[label]
        class_folder = os.path.join(processed_dir, split, class_name)
        
        # Ensure the folder exists
        os.makedirs(class_folder, exist_ok=True)

        # Convert back to uint8 and save the image
        image = (image * 255).astype(np.uint8)
        image_name = f"{split}_{class_name}_{i}.jpg"
        image_path = os.path.join(class_folder, image_name)
        cv2.imwrite(image_path, image)

if __name__ == "__main__":
    create_folder_structure()  # Create the directories
    preprocess_data()  # Process images and save them into respective folders
