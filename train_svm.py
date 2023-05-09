import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from joblib import dump

# Load the training and testing sets from the file
with np.load('train_test_split.npz') as data:
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

# Train the linear SVM
svm = LinearSVC(C=1.0, max_iter=1000, random_state=42)
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Calculate the accuracy score of the model
accuracy = balanced_accuracy_score(y_test, y_pred)
print('SVM Accuracy: {:.2f}%'.format(accuracy*100))

# Save the model
dump(svm, 'linear_svm_model.joblib')

# ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.