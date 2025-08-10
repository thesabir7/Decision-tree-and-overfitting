import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 1️⃣ Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2️⃣ Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Train Decision Tree without constraints (may overfit)
clf_no_limit = DecisionTreeClassifier(random_state=42)
clf_no_limit.fit(X_train, y_train)

# 4️⃣ Train Decision Tree with max_depth constraint
clf_limited = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_limited.fit(X_train, y_train)

# 5️⃣ Predictions
y_pred_no_limit = clf_no_limit.predict(X_test)
y_pred_limited = clf_limited.predict(X_test)

# 6️⃣ Accuracy scores
acc_no_limit = accuracy_score(y_test, y_pred_no_limit)
acc_limited = accuracy_score(y_test, y_pred_limited)

print("📊 Decision Tree Performance:")
print(f"No depth limit accuracy: {acc_no_limit:.4f}")
print(f"Max depth = 4 accuracy: {acc_limited:.4f}")

# 7️⃣ Visualize the limited depth tree
plt.figure(figsize=(16, 10))
plot_tree(
    clf_limited,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree (max_depth=4)")
plt.show()
