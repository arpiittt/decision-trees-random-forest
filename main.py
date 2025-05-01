import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

df = pd.read_csv("heart.csv")

print(df.head())
print(df.info())
print(df.describe())

x = df.drop("target", axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69)

dt = DecisionTreeClassifier(random_state=69)
dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

plt.figure(figsize=(40,30))
plot_tree(dt, filled=True, feature_names=x.columns, class_names=["No Disease", "Disease"])
plt.title("Decision Tree")
plt.savefig("decision_tree.png")

dt_limited = DecisionTreeClassifier(max_depth=4, random_state=69)
dt_limited.fit(x_train, y_train)
y_pred_limited = dt_limited.predict(x_test)
print("\nLimited Decision Tree Accuracy:", accuracy_score(y_test, y_pred_limited))

rf = RandomForestClassifier(n_estimators=100, random_state=69)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

importances = rf.feature_importances_
features = pd.Series(importances, index=x.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=features, y=features.index)
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance_plot.png")

cv_scores_dt = cross_val_score(dt, x, y, cv=5)
cv_scores_rf = cross_val_score(rf, x, y, cv=5)

print("\nDecision Tree CV Accuracy:", np.mean(cv_scores_dt))
print("Random Forest CV Accuracy:", np.mean(cv_scores_rf))

print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

