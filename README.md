# 📘 Day 5: Decision Trees & Overfitting

As part of my **#10DaysOfML** journey, today I implemented **Decision Trees** – a simple yet powerful algorithm for both classification and regression.

---

## 📌 What is a Decision Tree?

A **Decision Tree** splits the data into smaller subsets based on feature values, forming a tree-like structure of decision rules.  
At each **node**, the algorithm chooses the **best split** that improves prediction accuracy according to a metric like **Gini impurity** or **Entropy**.

---

## ✅ Advantages
- Easy to **understand and interpret** (can be visualized)
- Works with **numerical and categorical** data
- **No need to scale** features

---

## ⚠️ Disadvantages
- **Prone to overfitting** (learning noise instead of patterns)
- **Sensitive** to small variations in data

---

## ✂️ Avoiding Overfitting
- **Pre-pruning**: Limit `max_depth`, `min_samples_split`, or `min_samples_leaf`
- **Post-pruning**: Remove unnecessary branches after training
- **Ensembles**: Use methods like **Random Forests** or **Gradient Boosting**

---

## 🧠 Learning from Day 5
> A model that perfectly fits the training data often performs poorly on unseen data.  
> Simpler, shallower trees tend to generalize better.

---

## 📊 Dataset
- **Breast Cancer Dataset** from `scikit-learn` (same as Day 4 for consistency)

---
