import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def create_custom_cnn(input_shape):
    inputs = layers.Input(input_shape)

    # Convolutional layers
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Fully connected layers
    flatten = layers.Flatten()(pool3)
    dense1 = layers.Dense(256, activation='relu')(flatten)
    dropout1 = layers.Dropout(0.5)(dense1)

    dense2 = layers.Dense(64, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.5)(dense2)

    # Output layer
    output = layers.Dense(2, activation='softmax')(dropout2)

    return models.Model(inputs=inputs, outputs=output)

input_shape = (256, 256, 3)  # Modify this based on your image size and number of channels
model = create_custom_cnn(input_shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load your dataset
# X_train, y_train: your training images and corresponding points of interest labels
# X_test, y_test: your test images and corresponding points of interest labels
# Make sure the data is normalized and properly shaped

# Train the model
epochs = 50  # You can change this value depending on your requirements
batch_size = 16  # You can change this value depending on your system's memory

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f'Test accuracy: {test_acc}')

# Save the model
model.save('custom_cnn.h5')
