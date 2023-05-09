import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc
from joblib import load
import matplotlib.pyplot as plt

# Load the training and testing sets from the file
with np.load('train_test_split.npz') as data:
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

# Load the model
lr = load('logistic_regression_model.joblib')

# Make predictions on the test set
y_scores = lr.predict_proba(X_test)[:, 1]
y_pred = lr.predict(X_test)

# Calculate the accuracy score of the model
accuracy = balanced_accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))

# Compute the false positive rate, true positive rate, and threshold values
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Calculate the AUC (Area Under the Curve) of the ROC curve
roc_auc = auc(fpr, tpr)
print('roc_auc: {:.2f}%'.format(roc_auc*100))

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve, Balanced_acc = {:.2f}'.format(accuracy))
plt.legend(loc='lower right')
plt.show()

