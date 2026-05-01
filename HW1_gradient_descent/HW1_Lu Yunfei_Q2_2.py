import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


np.random.seed(42)
random.seed(42)


def load_iris():

    df = pd.read_csv("Iris.csv")

    X = df[["SepalLengthCm", "SepalWidthCm",
            "PetalLengthCm", "PetalWidthCm"]].values

    label_mapping = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }
    y = np.array([label_mapping[label] for label in df["Species"].values])
    return X, y

def split_train_test(X, y, test_ratio=0.2):
    total_samples = X.shape[0]
    test_samples = int(total_samples * test_ratio)

    test_indices = random.sample(range(total_samples), test_samples)
    train_indices = [i for i in range(total_samples) if i not in test_indices]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


def normalize_train_test(X_train, X_test):

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test

def add_bias(X):

    return np.hstack([np.ones((X.shape[0], 1)), X])

def one_hot(y, num_classes=3):

    y_onehot = np.zeros((len(y), num_classes))
    y_onehot[np.arange(len(y)), y] = 1
    return y_onehot


def softmax_function(z):

    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(X, y_true, weights, reg_type="none", lambda_reg=0.0):
    n_samples = len(y_true)
    y_pred = softmax_function(X @ weights)

    cross_entropy = -np.sum(y_true * np.log(y_pred + 1e-8)) / n_samples

    if reg_type == "L2":
        reg_term = (lambda_reg / 2) * np.sum(weights[1:] ** 2)
    elif reg_type == "L1":
        reg_term = lambda_reg * np.sum(np.abs(weights[1:]))
    else:
        reg_term = 0.0
    return cross_entropy + reg_term

def compute_gradient(X, y_true, weights, reg_type="none", lambda_reg=0.0):
    n_samples = len(y_true)
    y_pred = softmax_function(X @ weights)

    grad = (X.T @ (y_pred - y_true)) / n_samples

    if reg_type == "L2":
        grad[1:] += lambda_reg * weights[1:]
    elif reg_type == "L1":
        grad[1:] += lambda_reg * np.sign(weights[1:])
    return grad

def train_softmax_regression(X_train, y_train_onehot,
                             X_test, y_test,
                             epochs=10000, lr=0.1,
                             reg_type="none", lambda_reg=0.0):
    n_features = X_train.shape[1]
    n_classes = y_train_onehot.shape[1]

    np.random.seed(42)
    weights = np.random.randn(n_features, n_classes) * 0.01
    train_losses = []


    for epoch in range(epochs):
        loss = compute_loss(X_train, y_train_onehot, weights, reg_type, lambda_reg)
        train_losses.append(loss)
        grad = compute_gradient(X_train, y_train_onehot, weights, reg_type, lambda_reg)
        weights -= lr * grad


    def accuracy(X, y):
        y_pred = softmax_function(X @ weights)
        preds = np.argmax(y_pred, axis=1)
        return np.mean(preds == y) * 100

    train_acc = accuracy(X_train, y_train)
    test_acc = accuracy(X_test, y_test)
    return train_losses, train_acc, test_acc, weights


def analyze_weight_sparsity(weights):

    non_bias_weights = weights[1:]
    return np.mean(np.abs(non_bias_weights) < 1e-4) * 100


if __name__ == "__main__":

    X, y = load_iris()
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train, X_test = normalize_train_test(X_train, X_test)
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)
    y_train_onehot = one_hot(y_train)

    print(f"the num of train: {X_train.shape[0]}")
    print(f"the num of test: {X_test.shape[0]}")
    print("-" * 60)


    print("2.2 (a) No Regularization")
    loss_none, train_acc_none, test_acc_none, w_none = train_softmax_regression(
        X_train, y_train_onehot,
        X_test, y_test,
        reg_type="none"
    )
    print(f"Training Accuracy: {train_acc_none:.2f}%")
    print(f"Test Accuracy    : {test_acc_none:.2f}%")
    print("-" * 60)

    plt.figure(figsize=(10, 6))
    plt.plot(loss_none)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss (No Regularization)")
    plt.xlim(0, 10000)
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(0, 10001, 2000))
    plt.yticks(np.arange(0, 1.01, 0.2))
    plt.grid()
    plt.savefig("Q2_2.png", dpi=300, bbox_inches="tight")
    plt.close()


    print("2.2 (b) L2 Regularization")
    l2_weights = {}
    for lam in [0.01, 0.1, 1]:
        _, train_acc, test_acc, w_l2 = train_softmax_regression(
            X_train, y_train_onehot,
            X_test, y_test,
            reg_type="L2",
            lambda_reg=lam
        )
        l2_weights[lam] = w_l2
        print(f"λ={lam}")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Test Accuracy    : {test_acc:.2f}%\n")
    print("-" * 60)


    print("2.2 (c) L1 Regularization")
    l1_weights = {}
    for lam in [0.01, 0.1, 1]:
        _, train_acc, test_acc, w_l1 = train_softmax_regression(
            X_train, y_train_onehot,
            X_test, y_test,
            reg_type="L1",
            lambda_reg=lam
        )
        l1_weights[lam] = w_l1
        print(f"λ={lam}")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Test Accuracy    : {test_acc:.2f}%\n")
    print("-" * 60)


    print("2.2 (d) Comparison Between L1 and L2 Regularization")
    for lam in [0.01, 0.1, 1]:
        l1_sparsity = analyze_weight_sparsity(l1_weights[lam])
        l2_sparsity = analyze_weight_sparsity(l2_weights[lam])
        print(f"λ = {lam}")
        print(f"L1  -> Sparsity: {l1_sparsity:.2f}%")
        print(f"L2  -> Sparsity: {l2_sparsity:.2f}%\n")


