# train_model.py
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# Dummy training data
X = np.array([[1], [2], [3], [4], [5]])   # hours studied
y = np.array([40, 50, 60, 70, 80])       # marks

model = LinearRegression()
model.fit(X, y)

# Save the model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
