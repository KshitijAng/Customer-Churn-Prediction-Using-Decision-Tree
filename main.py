# 1. Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

warnings.filterwarnings('ignore')

# 2. Loading the dataset
df = pd.read_csv('dataset.csv')

# 2.1 Feature Engineering: Add new features to enrich the dataset
df['BalancePerProduct'] = df['Balance'] / (df['NumOfProducts'] + 1)  # avoid division by zero

# Features and target
x = df[['Age','NumOfProducts','Balance','BalancePerProduct']]
y = df[['Exited']]

# 3. Train-Test Split
# 70% of the data is used for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# 4. Training the Decision Tree model
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Determine Important features
# importances = clf.feature_importances_

# for feature, importance in zip(x.columns, importances):
#     print(f"{feature}: {importance:.4f}")

# 5. Prediction & Evaluation
# Accuracy is the proportion of correct predictions among the total number of cases processed
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Visualizing the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=['Age', 'NumOfProducts', 'Balance','BalancePerProduct'], class_names=['No Churn', 'Churn'])
plt.title('Decision Tree for Predicting Customer Churn')
plt.show()