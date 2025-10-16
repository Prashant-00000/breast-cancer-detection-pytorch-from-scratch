import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)

class MySimpleNN:
    def __init__(self, X):
        self.weights = torch.rand(X.shape[1], 1, dtype=torch.float32, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def forward(self, X):
        z = torch.matmul(X, self.weights) + self.bias
        y_pred = torch.sigmoid(z)
        return y_pred

    def loss_function(self, y_pred, y):
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        return -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()

learning_rate = 0.1
epochs = 25

model = MySimpleNN(X_train_tensor)

for epoch in range(epochs):
    y_pred = model.forward(X_train_tensor)
    loss = model.loss_function(y_pred, y_train_tensor)
    loss.backward()

    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad
        model.weights.grad.zero_()
        model.bias.grad.zero_()

    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

with torch.no_grad():
    y_pred_test = model.forward(X_test_tensor)
    y_pred_class = (y_pred_test > 0.5).float()
    correct = (y_pred_class == y_test_tensor).sum().item()
    accuracy = (correct / y_test_tensor.shape[0]) * 100

print(f"\nModel Accuracy: {accuracy:.2f}%")
print(f"Final Bias: {model.bias.item():.4f}")
