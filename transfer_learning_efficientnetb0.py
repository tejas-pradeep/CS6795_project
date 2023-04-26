import tensorflow as tf
from keras import layers, models
from keras.layers import preprocessing, RandomFlip, RandomRotation, RandomZoom
from keras.applications import EfficientNetB0
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras_tuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import numpy as np
from matplotlib import pyplot as plt
import pickle
import cv2

# Load your dataset here
# Replace this with your actual dataset loading function
def load_dataset():
    with open('C:\\Users\\rptej\\PycharmProjects\\CS6795_project\\project\\src\\data\\images_noNaN.pkl', 'rb') as f:
        data = pickle.load(f)

    # Filter images with the correct shape and their corresponding points of interest
    valid_data = [(row['image'], (row['first_x_fixation'], row['first_y_fixation']))
                  for _, row in data.iterrows() if row['image'].shape == (960, 1280, 3)]

    # Separate images and points of interest into two separate arrays
    images, points_of_interest = zip(*valid_data)
    images = np.stack(images, axis=0)
    points_of_interest = np.array(points_of_interest)

    return images, points_of_interest



# Pre-process the images and points of interest
def preprocess_data(images, points_of_interest):
    # Resize and normalize images
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        resized_images.append(resized_img)
    resized_images = np.array(resized_images).astype('float32') / 255.0

    return resized_images, points_of_interest
# Split the dataset into training and testing sets
def split_data(images, points_of_interest):
    x_train, x_test, y_train, y_test = train_test_split(images, points_of_interest, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

# Create the transfer learning model
# Modify the create_model function to accept hyperparameters
def create_model(hp):
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(480, 640, 3))
    
    # Unfreeze the last few layers
    num_unfrozen_layers = hp.Int('num_unfrozen_layers', min_value=1, max_value=15)
    for layer in base_model.layers[-num_unfrozen_layers:]:
        layer.trainable = True

    model = models.Sequential()

    # Add data augmentation
    # data_augmentation = tf.keras.Sequential([
    #     RandomFlip("horizontal_and_vertical"),
    #     RandomRotation(0.2),
    #     RandomZoom(0.2),
    # ])
    # model.add(data_augmentation)

    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    
    for _ in range(hp.Int('num_dense_layers', min_value=1, max_value=10)):
        model.add(layers.Dense(hp.Int('units', min_value=32, max_value=256, step=32), activation='relu'))
        model.add(layers.Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(layers.Dense(2, activation='linear'))

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')), loss='mse', metrics=['mae'])
    return model
    

def main():
    # Load and preprocess the dataset
    images, points_of_interest = load_dataset()
    images, points_of_interest = preprocess_data(images, points_of_interest)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = split_data(images, points_of_interest)

    # Initialize the Keras Tuner
    tuner = RandomSearch(
        create_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='efficientnetb0_points_of_interest'
    )

    # Callbacks for early stopping and learning rate reduction
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

   # Search for the best hyperparameters
    tuner.search(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping, reduce_lr])

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    best_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping, reduce_lr])

      # Evaluate the best model
    mse, mae = best_model.evaluate(x_test, y_test)

    print(f"\nMean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

     # Make predictions
    predictions = best_model.predict(x_test)

    # Display a few test images with actual and predicted fixations
    num_images_to_display = 5
    for i in range(num_images_to_display):
        plt.figure()
        plt.imshow(x_test[i])
        plt.scatter(y_test[i, 0] * x_test.shape[2], y_test[i, 1] * x_test.shape[1], c='r', marker='o', label='Actual')
        plt.scatter(predictions[i, 0] * x_test.shape[2], predictions[i, 1] * x_test.shape[1], c='b', marker='x', label='Predicted')
        plt.legend()
        plt.show()




    # Save the model using the SavedModel format
    # best_model.save("efficientnetb0_points_of_interest_savedmodel")

if __name__ == '__main__':
    main()
 