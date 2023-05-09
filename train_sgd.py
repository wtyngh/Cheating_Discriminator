import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from joblib import dump

# Load the training and testing sets from the file
with np.load('train_test_split.npz') as data:
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

# Train the SGD classifier
sgd = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train, y_train)

# Make predictions on the test set
y_pred = sgd.predict(X_test)

# Calculate the accuracy score of the model
accuracy = balanced_accuracy_score(y_test, y_pred)
print('SGD Accuracy: {:.2f}%'.format(accuracy*100))

# Save the model
dump(sgd, 'sgd_model.joblib')
