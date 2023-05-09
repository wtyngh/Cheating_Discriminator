import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from joblib import dump

# Load the training and testing sets from the file
with np.load('train_test_split.npz') as data:
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

# Train the logistic regression model
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Calculate the accuracy score of the model
accuracy = balanced_accuracy_score(y_test, y_pred)
print('Logistic Regression Accuracy: {:.2f}%'.format(accuracy*100))

# Save the model
dump(lr, 'logistic_regression_model.joblib')