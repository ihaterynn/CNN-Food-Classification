import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import cv2

# Path to the saved model
model_path = r"C:\Users\User\OneDrive\Desktop\Computational Intelligence\Assignment 2\CNN src\models\food11_cnn_model.h5"

# Path to the test dataset
processed_dir = r"C:\Users\User\OneDrive\Desktop\Computational Intelligence\Assignment 2\Malaysian-11 Food Dataset\processed_images"
test_dir = processed_dir + '/test'

# Load the model
model = tf.keras.models.load_model(model_path)

# Load the test dataset
def load_test_data():
    """Loads the test dataset."""
    img_size = (224, 224)
    batch_size = 20  

    test_gen = tf.keras.preprocessing.image_dataset_from_directory(
        directory=test_dir,
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False  
    )

    return test_gen

# Function to predict using the trained model
def predict(model, test_gen):
    """Predicts the classes for the test dataset."""
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=-1)
    return predictions, predicted_classes

# Visualize Predictions
def visualize_predictions(test_gen, predictions, predicted_classes, true_labels, class_names):
    """Visualizes the predictions along with true labels for correctly classified images."""
    correct_indices = np.where(predicted_classes == true_labels)[0]
    total_correct = len(correct_indices)

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.ravel()

    for i, idx in enumerate(np.random.choice(correct_indices, 25, replace=False)):
        img_path = test_gen.file_paths[idx]
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)

        axes[i].imshow(img_array.astype('uint8')) 
        axes[i].set_title("Pred: %s\nTrue: %s" % (
            class_names[predicted_classes[idx]], class_names[true_labels[idx]]))
        axes[i].axis('off')

    plt.suptitle(f"{total_correct} out of {len(true_labels)} images classified correctly")
    plt.subplots_adjust(wspace=1)
    plt.show()

# Classification Report
def classification_report_table(true_labels, predicted_classes, class_names):
    """Generates the classification report."""
    report = classification_report(true_labels, predicted_classes, target_names=class_names)
    print("Classification Report:\n", report)

# Visualize Confusion Matrix
def plot_confusion_matrix(predicted_classes, true_labels, class_names):
    """Plots confusion matrix for the predicted vs true labels."""
    cm = confusion_matrix(true_labels, predicted_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plot AUC-ROC Curve
def plot_auc_roc_curve(predictions, true_labels, class_names):
    """Plots the AUC-ROC curve for each class."""
    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()
    true_labels_bin = lb.fit_transform(true_labels)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    test_gen = load_test_data()
    predictions, predicted_classes = predict(model, test_gen)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_gen], axis=0)
    class_names = test_gen.class_names

    # Visualize predictions and metrics
    visualize_predictions(test_gen, predictions, predicted_classes, true_labels, class_names)
    classification_report_table(true_labels, predicted_classes, class_names)
    plot_confusion_matrix(predicted_classes, true_labels, class_names)
    plot_auc_roc_curve(predictions, true_labels, class_names)
