import tensorflow as tf
import os
import sys
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt


sys.path.append(r'C:\Users\User\OneDrive\Desktop\Computational Intelligence\Assignment 2\CNN src')
from models.cnn import create_cnn_model  

# Define directory paths
processed_dir = r"C:\Users\User\OneDrive\Desktop\Computational Intelligence\Assignment 2\Malaysian-11 Food Dataset\processed_images"

# Load preprocessed datasets
def load_datasets():
    """Loads preprocessed datasets from directories."""
    batch_size = 25
    img_size = (224, 224)
    
    # Load training data
    train_gen = tf.keras.preprocessing.image_dataset_from_directory(
        directory=processed_dir + '/train', 
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Load validation data
    val_gen = tf.keras.preprocessing.image_dataset_from_directory(
        directory=processed_dir + '/validate',  
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_gen, val_gen


# Data Augmentation - Enhanced
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),  
    tf.keras.layers.RandomRotation(0.2),  
    tf.keras.layers.RandomZoom(0.2), 
    tf.keras.layers.RandomContrast(0.2),  
    tf.keras.layers.RandomBrightness(0.2),  
    tf.keras.layers.RandomHeight(0.2),  
    tf.keras.layers.RandomWidth(0.2),  
])

# Function to plot training accuracy over epochs
def plot_training_accuracy(history):
    """Plots training accuracy over epochs."""
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
    else:
        print("No training history found for accuracy.")

if __name__ == "__main__":
    # Load datasets
    train_gen, val_gen = load_datasets()

    # Create CNN model (now sourced from cnn.py)
    model = create_cnn_model()

    # Compile the model with Adam optimizer and ReduceLROnPlateau
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks: ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # Train the model 
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,  
        verbose=1, 
        callbacks=[reduce_lr]
    )

    # Save the model
    save_path = r"C:\Users\User\OneDrive\Desktop\Computational Intelligence\Assignment 2\CNN src\models\food11_cnn_model.h5"
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # Visualize the training accuracy over epochs
    plot_training_accuracy(history)
