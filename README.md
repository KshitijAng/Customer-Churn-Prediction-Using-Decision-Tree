# 📊 Customer Churn Prediction Using Decision Tree

## Overview

Retaining a customer is significantly more cost-effective than aquiring a new one. This project aims to predict customer churn using a Decision Tree classifier. By analyzing customer data, 
the model identifies patterns that indicate whether a customer is likely to leave, enabling businesses to implement proactive retention strategies.

## Brief on Decision Tree
A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. 
It mimics human decision-making by splitting data into branches based on feature values, forming a tree-like structure.

**Key Concepts**
- Root Node: The top-most node that represents the initial feature split.
- Decision Nodes: Nodes where the dataset is split based on a condition.
- Leaf Nodes: Final outcomes or predictions (e.g., class labels).
- Branches: Paths connecting nodes that represent decision rules.

**Pros**
- Easy to understand and interpret.
- Requires little data preprocessing.
- Handles both numerical and categorical features.

**Cons**
- Prone to overfitting, especially with deep trees.
- Less accurate than ensemble methods like Random Forests or Gradient Boosting.


## 📁 Dataset

The dataset comprises customer information, including:

- **Age**: Customer's age
- **Number of Products**: Number of products the customer has with the company
- **Balance**: Account balance
- **Exited**: Target variable indicating if the customer has churned (1) or not (0)

*Note: Ensure the dataset is named `dataset.csv` and placed in the project directory.*

## 🧠 Features and Target

- **Features (X)**:
  - Age
  - Number of Products
  - Balance
  - Balance per Product (engineered feature)
  - Age Bin (categorized age groups)

- **Target (y)**:
  - Exited (1 if the customer churned, 0 otherwise)

## 🛠️ Implementation Steps

1. **Import Libraries**: Utilize essential libraries such as `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.

2. **Load Dataset**: Read the `dataset.csv` file into a pandas DataFrame.

3. **Feature Engineering**:
   - Create a new feature `BalancePerProduct` by dividing Balance by (Number of Products + 1).
   - Categorize Age into bins to create an `AgeBin` feature.

4. **Data Preprocessing**:
   - Convert categorical variables into dummy/indicator variables.
   - Split the dataset into training and testing sets (70% training, 30% testing).

5. **Model Training**:
   - Initialize a Decision Tree classifier with a maximum depth to prevent overfitting.
   - Train the model using the training data.

6. **Model Evaluation**:
   - Predict on the test set.
   - Calculate accuracy score.
   - Generate a classification report.
     
7. **Visualization**:
   - Display the structure of the trained Decision Tree.
   - Plot feature importances.

## 📈 Results

- **Model Accuracy**: Improved model accuracy from 70% to 84% on the test set through iterative feature engineering and tuning.
- **Insights**:
  - Engineered `BalancePerProduct` feature which contributed to improved model performance.
  - The Decision Tree visualization provides interpretability into how decisions are made.

--
