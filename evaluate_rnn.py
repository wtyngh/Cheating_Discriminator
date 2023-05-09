import numpy as np
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load the training and testing sets from the file
with np.load('train_test_split.npz') as data:
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

# Reshape the input data to a 4D tensor (batch_size, timesteps, input_dim, channels)
X_train = X_train.reshape((-1, 30, 192, 5))
X_test = X_test.reshape((-1, 30, 192, 5))

# Reshape the input data to a 3D tensor (batch_size, timesteps, input_dim)
X_train = np.transpose(X_train, (0, 2, 1, 3))
X_train = X_train.reshape((-1, 192, 150))
X_test = np.transpose(X_test, (0, 2, 1, 3))
X_test = X_test.reshape((-1, 192, 150))

# Load the model
model = keras.models.load_model('rnn_model.h5')

# Make predictions on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob).flatten()

# Calculate the accuracy score of the model
accuracy = balanced_accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))

# Compute the false positive rate, true positive rate, and threshold values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate the AUC (Area Under the Curve) of the ROC curve
roc_auc = auc(fpr, tpr)
print('roc_auc: {:.2f}%'.format(roc_auc*100))

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RNN ROC Curve, Balanced_acc = {:.2f}'.format(accuracy))
plt.legend(loc='lower right')
plt.show()

