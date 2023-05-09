import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
cheaters_X = np.load('cheaters/cheaters.npy')
legit_X = np.load('legit/legit.npy')
data_X = np.concatenate((cheaters_X, legit_X))
cheaters_N = cheaters_X.shape[0]
legit_N = legit_X.shape[0]
data_y = np.concatenate((np.ones((cheaters_N,)), np.zeros((legit_N,))))

# Split the data into features and target variables
X = data_X.reshape(cheaters_N + legit_N, -1)
print(f"cheaters_N:{cheaters_N}, legit_N:{legit_N}")
print(f"X shape:{X.shape}")
y = data_y

# Scale the input features using standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save the training and testing sets to a file
np.savez('train_test_split.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# Load the training and testing sets from the file
with np.load('train_test_split.npz') as data:
    X_train_loaded = data['X_train']
    X_test_loaded = data['X_test']
    y_train_loaded = data['y_train']
    y_test_loaded = data['y_test']

# Check that the loaded data matches the original data
print('X_train loaded matches X_train:', np.array_equal(X_train, X_train_loaded))
print('X_test loaded matches X_test:', np.array_equal(X_test, X_test_loaded))
print('y_train loaded matches y_train:', np.array_equal(y_train, y_train_loaded))
print('y_test loaded matches y_test:', np.array_equal(y_test, y_test_loaded))