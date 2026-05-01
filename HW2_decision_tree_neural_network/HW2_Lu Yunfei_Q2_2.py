# 1. 导入所有需要的库
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# Step A
class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(30, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)
        self.fc3 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        hidden_2d = self.fc2(x)
        out = self.fc3(hidden_2d)
        out = self.sigmoid(out)
        return out, hidden_2d

# Step B
model = CustomMLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 200
train_loss_list = []

for epoch in range(epochs):
    model.train()
    outputs, _ = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss_list.append(loss.item())


plt.figure(figsize=(8, 4))
plt.plot(train_loss_list, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# Step C
model.eval()
with torch.no_grad():
    test_outputs, test_hidden_2d = model(X_test)
    predictions = (test_outputs >= 0.5).float()
    accuracy = (predictions == y_test).sum().item() / len(y_test)
    print(f'Test Accuracy: {accuracy*100:.2f}%')


hidden_features = test_hidden_2d.numpy()
labels = y_test.numpy().flatten()

plt.figure(figsize=(8, 6))

plt.scatter(hidden_features[labels==0, 0], hidden_features[labels==0, 1],
            color='red', label='Malignant (Class 0)', alpha=0.7, edgecolors='k')

plt.scatter(hidden_features[labels==1, 0], hidden_features[labels==1, 1],
            color='blue', label='Benign (Class 1)', alpha=0.7, edgecolors='k')

plt.title("2D Hidden Space Learned by MLP")
plt.xlabel("Hidden Neuron 1 Output")
plt.ylabel("Hidden Neuron 2 Output")
plt.legend()
plt.grid(True)
plt.show()