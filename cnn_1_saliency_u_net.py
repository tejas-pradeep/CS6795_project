import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def create_saliency_unet(input_shape):
    inputs = layers.Input(input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Decoder
    up3 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv2), conv1], axis=-1)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

    # Output layer
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv3)
    
    return models.Model(inputs=inputs, outputs=output)

input_shape = (256, 256, 3)  # Modify this based on your image size and number of channels
model = create_saliency_unet(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load your dataset
# X_train, y_train: your training images and corresponding points of interest maps
# X_test, y_test: your test images and corresponding points of interest maps
# Make sure the data is normalized and properly shaped

# Train the model
epochs = 50  # You can change this value depending on your requirements
batch_size = 16  # You can change this value depending on your system's memory

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f'Test accuracy: {test_acc}')

# Save the model
model.save('saliency_unet.h5')
