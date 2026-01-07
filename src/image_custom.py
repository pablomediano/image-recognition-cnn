"""
Image Recognition using Convolutional Neural Networks (CNN)
Custom dataset classification
"""

# Import required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)

# ==========================================
# CONFIGURATION
# ==========================================

# Set your dataset path here
DATASET_PATH = 'dataset'  # Change this to your dataset folder path
IMAGE_SIZE = (128, 128)   # Target size for all images
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2    # 20% of data for validation
TEST_SPLIT = 0.1          # 10% of data for testing

# ==========================================
# OUTPUT DIRECTORIES
# ==========================================

RESULTS_DIR = "results"
SAMPLES_DIR = os.path.join(RESULTS_DIR, "sample_images")
HISTORY_DIR = os.path.join(RESULTS_DIR, "training_history")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# ==========================================
# PART 1: LOAD CUSTOM DATASET FROM FOLDERS
# ==========================================

def load_custom_dataset(dataset_path, img_size=(128, 128)):
    """
    Load images from folder structure where each subfolder is a class
    
    Args:
        dataset_path: Path to dataset root folder
        img_size: Target size for resizing images
    
    Returns:
        images: numpy array of images
        labels: numpy array of labels
        class_names: list of class names
    """
    print(f"\n=== Loading Custom Dataset from {dataset_path} ===")
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' not found!")
    
    # Get all subdirectories (class names)
    class_names = sorted([d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))])
    
    if len(class_names) == 0:
        raise ValueError(f"No subdirectories found in {dataset_path}. "
                        "Please organize images into folders by class.")
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    images = []
    labels = []
    
    # Supported image formats
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    # Load images from each class folder
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if os.path.splitext(f.lower())[1] in valid_extensions]
        
        print(f"Loading {len(image_files)} images from '{class_name}'...")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  Warning: Could not read {img_path}")
                    continue
                
                # Convert BGR to RGB (OpenCV loads as BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize image
                img = cv2.resize(img, img_size)
                
                images.append(img)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
                continue
    
    print(f"\nSuccessfully loaded {len(images)} images total")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels, class_names

# Load the dataset
try:
    images, labels, class_names = load_custom_dataset(DATASET_PATH, IMAGE_SIZE)
except Exception as e:
    print(f"\nError loading dataset: {e}")
    print("\nPlease ensure your dataset is organized as follows:")
    print("dataset/")
    print("    ├── class1/")
    print("    │   ├── image1.jpg")
    print("    │   └── image2.jpg")
    print("    ├── class2/")
    print("    │   └── image1.jpg")
    print("    └── ...")
    raise

# Print dataset information
print(f"\nDataset Summary:")
print(f"Total images: {len(images)}")
print(f"Image shape: {images.shape[1:]} (Height x Width x Channels)")
print(f"Number of classes: {len(class_names)}")
print(f"Classes: {class_names}")

# Print class distribution
unique, counts = np.unique(labels, return_counts=True)
print("\nClass Distribution:")
for class_idx, count in zip(unique, counts):
    print(f"  {class_names[class_idx]:15s}: {count} images")

# ==========================================
# VISUALIZE SAMPLE IMAGES
# ==========================================

def plot_sample_images(images, labels, class_names, num_samples=15):
    """Display a grid of sample images with their labels"""
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(indices):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[idx])
        plt.title(class_names[labels[idx]], fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLES_DIR, 'sample_images.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Sample images saved as 'sample_images.png'")

print("\n=== Displaying Sample Images ===")
plot_sample_images(images, labels, class_names)

# ==========================================
# PART 2: SPLIT AND PREPROCESS THE DATA
# ==========================================

print("\n=== Splitting Dataset ===")

# First split: separate test set
x_temp, x_test, y_temp, y_test = train_test_split(
    images, labels, 
    test_size=TEST_SPLIT, 
    random_state=42,
    stratify=labels  # Ensure balanced classes
)

# Second split: separate train and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp,
    test_size=VALIDATION_SPLIT/(1-TEST_SPLIT),
    random_state=42,
    stratify=y_temp
)

print(f"Training images: {len(x_train)}")
print(f"Validation images: {len(x_val)}")
print(f"Test images: {len(x_test)}")

# Normalize pixel values to range [0, 1]
print("\n=== Preprocessing Data ===")
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"Pixel value range after normalization: [{x_train.min():.2f}, {x_train.max():.2f}]")

# Convert labels to categorical (one-hot encoding)
num_classes = len(class_names)
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# ==========================================
# DATA AUGMENTATION (Optional but Recommended)
# ==========================================

print("\n=== Setting Up Data Augmentation ===")

# Create data augmentation generator for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# No augmentation for validation/test data
val_datagen = ImageDataGenerator()

# Create generators
train_generator = train_datagen.flow(
    x_train, y_train_cat,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_generator = val_datagen.flow(
    x_val, y_val_cat,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Data augmentation enabled for training set")

# ==========================================
# PART 3: BUILD THE CNN MODEL
# ==========================================

print("\n=== Building the Model ===")

# Adjust model complexity based on dataset size
if len(images) < 1000:
    # Simpler model for smaller datasets
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                      input_shape=(*IMAGE_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
else:
    # More complex model for larger datasets
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                      input_shape=(*IMAGE_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

# Display model architecture
model.summary()

# ==========================================
# PART 4: COMPILE THE MODEL
# ==========================================

print("\n=== Compiling the Model ===")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# PART 5: TRAIN THE MODEL
# ==========================================

print("\n=== Training the Model ===")
print("This may take several minutes...")

# Calculate steps per epoch
steps_per_epoch = len(x_train) // BATCH_SIZE
validation_steps = len(x_val) // BATCH_SIZE

# Add early stopping and model checkpoint callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Train the model with data augmentation
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete!")

# ==========================================
# PART 6: EVALUATE THE MODEL
# ==========================================

print("\n=== Evaluating Model Performance ===")
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# ==========================================
# VISUALIZE TRAINING HISTORY
# ==========================================

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(HISTORY_DIR, 'training_history.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Training history saved as 'training_history.png'")

plot_training_history(history)

# ==========================================
# PART 7: TEST PREDICTIONS
# ==========================================

print("\n=== Making Predictions on Test Images ===")

# Make predictions on test set
num_test_samples = min(100, len(x_test))
predictions = model.predict(x_test[:num_test_samples])

def plot_predictions(images, true_labels, predictions, class_names, num_samples=12):
    """Display predictions vs actual labels"""
    num_samples = min(num_samples, len(images))
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i])
        
        predicted_class = np.argmax(predictions[i])
        true_class = true_labels[i]
        confidence = predictions[i][predicted_class] * 100
        
        # Color code: green if correct, red if incorrect
        color = 'green' if predicted_class == true_class else 'red'
        
        plt.title(f"True: {class_names[true_class]}\n"
                 f"Pred: {class_names[predicted_class]} ({confidence:.1f}%)",
                 color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTIONS_DIR, 'predictions.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Predictions saved as 'predictions.png'")

plot_predictions(x_test, y_test, predictions, class_names)

# ==========================================
# PART 8: CALCULATE PER-CLASS ACCURACY
# ==========================================

print("\n=== Per-Class Accuracy ===")
predictions_all = model.predict(x_test, verbose=0)
predicted_classes = np.argmax(predictions_all, axis=1)

for i, class_name in enumerate(class_names):
    # Find all instances of this class
    class_mask = y_test == i
    if np.sum(class_mask) > 0:
        class_accuracy = np.mean(predicted_classes[class_mask] == i)
        print(f"{class_name:20s}: {class_accuracy * 100:.2f}% "
              f"({np.sum(class_mask)} test images)")
    else:
        print(f"{class_name:20s}: No test images")

# ==========================================
# OPTIONAL: SAVE THE MODEL
# ==========================================

print("\n=== Saving Model ===")
model.save(os.path.join(RESULTS_DIR, 'custom_image_recognition_model.keras'))
print("Model saved as 'custom_image_recognition_model.keras'")

# Save class names for later use
with open(os.path.join(RESULTS_DIR, 'class_names.txt'), 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
print("Class names saved as 'class_names.txt'")

