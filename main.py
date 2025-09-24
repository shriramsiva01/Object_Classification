import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define constants
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 10
categories = ["Bikes", "Cars"]  # Category names

# Set paths to data folders
train_dir = 'C:\Project\cars and bikes classifier\Dataset\Train'  # Adjust these paths according to your dataset location
test_dir = 'C:\Project\cars and bikes classifier\Dataset\Test'

# Preprocess the images using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    epochs=EPOCHS,
                    validation_data=test_generator)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)
print("Test accuracy:", accuracy)

# Get class indices
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}

# Print category, object, and its accuracy
for label_index, class_accuracy in enumerate(history.history['accuracy']):
    if label_index < len(categories):
        category_name = categories[label_index]
        class_name = class_labels[label_index]
        print(f"Category: {category_name}, Object: {class_name}, Accuracy: {class_accuracy}")

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict an image from a file path
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    class_label = class_labels[predicted_class]
    print(f"Predicted Class: {class_label}")
    print(f"Confidence: {confidence}")

# Example usage
# Replace 'path_to_your_image.jpg' with the actual path to your image
image_path = "C:\Project\cars and bikes classifier\Dataset\Test\Cars\df.jpg"  # Replace with your image path
predict_image(image_path)
