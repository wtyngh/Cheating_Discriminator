import os
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

import numpy as np
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
import random

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1)
   tf.random.set_seed(1)
   np.random.seed(1)
   random.seed(1)

# For reproducible results
reset_random_seeds()

# Load the training and testing sets from the file
with np.load('train_test_split.npz') as data:
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob).flatten()

# Calculate the accuracy score of the model
accuracy = balanced_accuracy_score(y_test, y_pred)
print('NN Accuracy: {:.2f}%'.format(accuracy*100))

# Save the model
model.save('neural_network_model.h5')
