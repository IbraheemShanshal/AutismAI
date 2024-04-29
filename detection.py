import os.path

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives, AUC
from keras.src.layers import BatchNormalization
from keras.src.optimizers import Adam
from numpy.random.mtrand import random
import random as rn
from keras.regularizers import l2
from sklearn.metrics import  roc_curve

# fix the seeds for consistency
Seed = 1
#Set random seed for TensorFlow
tf.random.set_seed(Seed)

# Set random seed for NumPy
np.random.seed(Seed)
rn.seed(Seed)

# preprocess the data
directory = 'data'  # directory of the dataset
image_extensions = 'jpg'  # the only extension in the dataset
# load the data into tensorflow
data = tf.keras.utils.image_dataset_from_directory(directory,
                                                   color_mode='grayscale',
                                                   batch_size = 32,
                                                   image_size=(365, 365))
print(data.class_names)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# scaling the data
scaled_data = data.map(lambda x, y :(x/255,y))
scaled_iterator = scaled_data.as_numpy_iterator() # rebatching after the data is scaled
batch = scaled_iterator.next()


#data split
total_size = len(list(data))
training_size = int(total_size * .75)
validation_size = int(total_size * .13)
testing_size = int(total_size * .12)

train = scaled_data.take(training_size)
test = scaled_data.skip(training_size).take(testing_size)
validate = scaled_data.skip(training_size+testing_size).take(validation_size)

#building the model
model = Sequential()

# CNN layers
model.add(Conv2D(4, (3,3), padding='same', activation = 'relu', kernel_regularizer=l2(0.001), input_shape=(365,365,1)))
model.add(BatchNormalization())
model.add(Conv2D(8, (3,3),  padding='same', activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3),  padding='same', activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(71, (3,3),  padding='same', activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D())



model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))     # this is to make the output either 0 or 1 for binary classification

learning_rate = 0.001
# Use the Adam optimizer with a specified learning rate
optimizer = Adam(learning_rate=learning_rate)

model.compile( optimizer = optimizer, loss = tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

print(model.summary())

# training
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(train, epochs=20, validation_data=validate)

# Gather predictions and actual labels
y_true = np.concatenate([y for x, y in test], axis=0)
y_scores = np.concatenate([model.predict(x) for x, y in test], axis=0)

# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold (based on Youden index): {optimal_threshold}")

# test performance
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
tp = TruePositives()
tn = TrueNegatives()
fp = FalsePositives()
fn = FalseNegatives()
auc = AUC()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    tp.update_state(y, yhat)
    tn.update_state(y, yhat)
    fp.update_state(y, yhat)
    fn.update_state(y, yhat)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
    auc.update_state(y, yhat)

sensitivity = tp.result().numpy() / (tp.result().numpy() + fn.result().numpy())
specificity = tn.result().numpy() / (tn.result().numpy() + fp.result().numpy())
npv = tn.result().numpy() / (tn.result().numpy() + fn.result().numpy())
ppv = tp.result().numpy() / (tp.result().numpy() + fp.result().numpy())
auc_value = auc.result().numpy()

# printing results
print(f"TP: {tp.result().numpy()}, TN: {tn.result().numpy()}, FP: {fp.result().numpy()}, FN: {fn.result().numpy()}")
print(f"Sens.: {sensitivity:.2f}, Spec.: {specificity:.2f}, NPV: {npv:.2f}, PPV: {ppv:.2f}, AUC: {auc_value:.2f}")
print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy: {acc.result().numpy()}')

#saving the model
model.save(os.path.join('models','autismAiModel.h5'))