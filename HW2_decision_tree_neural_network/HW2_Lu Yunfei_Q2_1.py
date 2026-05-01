import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# Global setting
# =========================
RANDOM_SEED = 42
TEST_SIZE = 0.2

# =========================
# Q2.1 Load dataset
# =========================
# [TASK 1] Load the breast cancer dataset
data = load_breast_cancer()

# [TASK 2] Get the feature matrix X and label vector y
X = data.data
y = data.target

# [TASK 3] Split the dataset into training and test sets
# Requirements:
# - test_size = TEST_SIZE
# - random_state = RANDOM_SEED
# - stratify = y
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

print("Training size:", len(X_train))
print("Test size:", len(X_test))

# =========================
# Q2.2 Decision Tree
# =========================
# [TASK 4] Build a Decision Tree classifier
# Requirements:
# - criterion = 'entropy'
# - max_depth = 3
# - random_state = RANDOM_SEED
dt_clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    random_state=RANDOM_SEED
)

# [TASK 5] Train the Decision Tree model
dt_clf.fit(X_train, y_train)

# [TASK 6] Predict on both training and test sets
y_train_pred_dt = dt_clf.predict(X_train)
y_test_pred_dt = dt_clf.predict(X_test)

# [TASK 7] Compute the training and test accuracy of the Decision Tree
train_acc_dt = accuracy_score(y_train, y_train_pred_dt)
test_acc_dt = accuracy_score(y_test, y_test_pred_dt)

print("\nDecision Tree Results")
print(f"Train accuracy: {train_acc_dt:.4f}")
print(f"Test accuracy: {test_acc_dt:.4f}")

# =========================
# Q2.3 Random Forest
# =========================
n_trees_list = [10, 50, 100]
rf_results = []

for n_trees in n_trees_list:
    # [TASK 8] Build a Random Forest classifier
    # Requirements:
    # - n_estimators = n_trees
    # - criterion = 'entropy'
    # - max_depth = 3
    # - max_features = 'sqrt'
    # - random_state = RANDOM_SEED
    rf_clf = RandomForestClassifier(
        n_estimators=n_trees,
        criterion='entropy',
        max_depth=3,
        max_features='sqrt',
        random_state=RANDOM_SEED
    )

    # [TASK 9] Train the Random Forest model
    rf_clf.fit(X_train, y_train)

    # [TASK 10] Predict on both training and test sets
    y_train_pred_rf = rf_clf.predict(X_train)
    y_test_pred_rf = rf_clf.predict(X_test)

    # [TASK 11] Compute the training and test accuracy
    train_acc_rf = accuracy_score(y_train, y_train_pred_rf)
    test_acc_rf = accuracy_score(y_test, y_test_pred_rf)

    rf_results.append((n_trees, train_acc_rf, test_acc_rf))

print("\nRandom Forest Results")
for n_trees, train_acc_rf, test_acc_rf in rf_results:
    print(f"n_estimators = {n_trees}: "
          f"train accuracy = {train_acc_rf:.4f}, "
          f"test accuracy = {test_acc_rf:.4f}")

# =========================
# Q2.4 Analysis
# =========================
# [TASK 12] Find the model with the best test accuracy and print the results
all_models = {
    "Decision Tree": test_acc_dt,
    "Random Forest(n_estimators=10)": rf_results[0][2],
    "Random Forest(n_estimators=50)": rf_results[1][2],
    "Random Forest(n_estimators=100)": rf_results[2][2]
}
best_model_name = max(all_models, key=all_models.get)
best_test_acc = all_models[best_model_name]

print("\nBest model:")
print("Model:", best_model_name)
print(f"Best test accuracy: {best_test_acc:.4f}")