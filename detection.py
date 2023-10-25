import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# check tenser flow and keras versions (for debugging purposes)
print(tf.__version__)
print(keras.__version__)



