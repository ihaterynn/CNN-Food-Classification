import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_cnn_model(input_shape=(224, 224, 3), num_classes=3):
    l2_reg = regularizers.l2(1e-3)

    model = models.Sequential([  
        # Input Layer
        layers.InputLayer(input_shape=input_shape),

        # Convolutional block 1
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2_reg),  
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolutional block 2
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2_reg), 
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolutional block 3
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flattening layer
        layers.Flatten(),

        # Fully-connected layer (Dense)
        layers.Dense(256, activation='relu', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  

        # Output layer
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2_reg)
    ])

    return model

if __name__ == "__main__":
    model = create_cnn_model()
    model.summary()