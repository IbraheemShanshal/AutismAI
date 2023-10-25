import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def preprocess_image(file_path, target_size=(150, 150)):
    # Load the image
    img = load_img(file_path, target_size=target_size)
    # Convert to array and preprocess
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array
