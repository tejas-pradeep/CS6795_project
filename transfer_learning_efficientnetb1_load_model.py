import tensorflow as tf
from keras import layers, models
from keras.layers import preprocessing, RandomFlip, RandomRotation, RandomZoom
from keras.applications import EfficientNetV2B1
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

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print('Using GPU:', physical_devices[0])
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print('No GPU found, using CPU instead.')
# Load your dataset here
# Replace this with your actual dataset loading function
def load_dataset():
    # with open('C:\\Users\\rptej\\PycharmProjects\\CS6795_project\\project\\src\\data\\images_noNaN.pkl', 'rb') as f:
    #     data = pickle.load(f)

    # # Filter images with the correct shape and their corresponding points of interest
    # valid_data = [(row['image'], (row['first_x_fixation'], row['first_y_fixation']))
    #               for _, row in data.iterrows() if row['image'].shape == (960, 1280, 3)]

    # # Separate images and points of interest into two separate arrays
    # images, points_of_interest = zip(*valid_data)
    # images = np.stack(images, axis=0)
    # points_of_interest = np.array(points_of_interest)

    # return images, points_of_interest
    images = np.load('images.npy')
    fixations = np.load('poi.npy')

    return images, fixations



# Pre-process the images and points of interest
def preprocess_data(images, points_of_interest):
    # Resize and normalize images
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        resized_images.append(resized_img)
    resized_images = np.array(resized_images).astype('float32') / 255.0

    # Scale points of interest
    original_image_size = np.array([1280, 960])
    new_image_size = np.array([640, 480])
    points_of_interest = (points_of_interest / original_image_size) * new_image_size

    return resized_images, points_of_interest

# Split the dataset into training and testing sets
def split_data(images, points_of_interest):
    x_train, x_test, y_train, y_test = train_test_split(images, points_of_interest, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

# Create the transfer learning model
# Modify the create_model function to accept hyperparameters
def load_model(model):
    if model == 1:
        return tf.keras.models.load_model('efficientnetb1_points_of_interest_savedmodel')
    else:
        return tf.keras.models.load_model('efficientnetb2_points_of_interest_savedmodel')

    

def main():
    # Load and preprocess the dataset
    images, points_of_interest = load_dataset()
    images, points_of_interest = preprocess_data(images, points_of_interest)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = split_data(images, points_of_interest)

    
    # Retrieve the best model
    best_model = load_model(2)


    

      # Evaluate the best model
    mse, mae = best_model.evaluate(x_test, y_test)

    print(f"\nMean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

  # Make predictions
    predictions = best_model.predict(x_test)

    # Display a few test images with actual and predicted fixations
    num_images_to_display = 5

    diff_array = np.abs(predictions - y_test)

    print('Diff summary.')
    print(f'Mean:{np.mean(diff_array, axis=0)}  Max:{np.max(diff_array, axis=0)}  Median:{np.median(diff_array, axis=0)}')

    # create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # plot a histogram of the first column in the left subplot
    axs[0].hist(diff_array[:, 0])
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Histogram of x vals')

    # plot a histogram of the second column in the right subplot
    axs[1].hist(diff_array[:, 1])
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram of y vals')

    # adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0.3)

    # # display the plot
    plt.show()
    # for i in range(num_images_to_display):
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    #     # Show the image with the actual point
    #     ax1.imshow(x_test[i])
    #     ax1.scatter(y_test[i, 0], y_test[i, 1], c='r', marker='o', label='Actual')
    #     ax1.set_title('Actual')
    #     ax1.legend()

    #     # Show the image with the predicted point
    #     ax2.imshow(x_test[i])
    #     ax2.scatter(predictions[i, 0], predictions[i, 1], c='b', marker='x', label='Predicted')
    #     ax2.set_title('Predicted')
    #     ax2.legend()

    #     plt.show()

    # print(predictions)

if __name__ == '__main__':
    main()
 