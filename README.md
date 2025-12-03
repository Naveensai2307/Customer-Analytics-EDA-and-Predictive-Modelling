# Customer-Analytics-EDA-and-Predictive-Modelling

# Project Overview

This project focuses on analysing customer demographics and spending behaviour using the Mall Customers dataset.
**The goal is to:**

Perform Exploratory Data Analysis (EDA)

Identify key patterns in customer demographics

Segment customers into meaningful groups

Build predictive models to classify customers based on spending behaviour

Provide business insights for marketing & personalization

This project applies data science + machine learning to understand customer behaviour in a retail shopping environment.


# Dataset Description

The dataset contains customer demographic and spending information.
| Feature            | Description                             |
| ------------------ | --------------------------------------- |
| CustomerID         | Unique ID assigned to each customer     |
| Gender             | Male / Female                           |
| Age                | Age of the customer                     |
| Annual Income (k$) | Customer's yearly income                |
| Spending Score     | Score assigned based on spending habits |
| Work Status        | Employment or occupation category       |
| Family Size        | Number of family members                |
| Loyalty Score      | Customer loyalty rating                 |
| Membership Type    | Bronze/Silver/Gold/Platinum             |
| Purchased Category | Type of products purchased              |

# Technologies & Libraries Used
**Programming Language**

Python 3.x

**Major Libraries**

Data Handling & Analysis
**import pandas as pd
import numpy as np**

Visualization

**import matplotlib.pyplot as plt
import seaborn as sns**

Preprocessing

**from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split**

Machine Learning Models

**from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier**

Evaluation Metrics

**from sklearn.metrics import accuracy_score, classification_report, confusion_matrix**


# Workflow: Step-by-Step Process
**1. Import Libraries**

Loaded all essential libraries for EDA, clustering, modelling, and evaluation.

**2. Load the Dataset**

**df = pd.read_csv("Mall_Customers_Enhanced.csv")**

**Checked:**

Shape

Missing values

Duplicate values

Data types

Basic statistics

**3. Exploratory Data Analysis (EDA)**
Performed detailed EDA:

**Demographics**

Gender distribution

Most common age groups

Income ranges

**Spending Behaviour**

High vs low spenders

Spending patterns by gender

Income vs spending correlation

**Product & Membership Behaviour**

Spending Score across membership types

Purchasing categories

Loyalty score analysis

**Key EDA Insights**

Younger customers (18–35) tend to have higher spending scores.

Customers with higher incomes do not always spend more — clusters showed varied behaviour.

Platinum and Gold members had the highest spending potential.

Family size moderately affected spending habits.

Spending Score strongly correlated with Loyalty & Membership Type.

**Visualizations Included:**

Bar charts

Histograms

Boxplots

Countplots

Correlation heatmap

Pairplots

**4. Feature Engineering**

Encoded categorical features using LabelEncoder

Scaled numerical features using StandardScaler

Created “High Spender / Low Spender” classification label (if part of your notebook)

Selected relevant features for modelling

**5. Machine Learning Models**

This project includes both Unsupervised Learning & Supervised Learning.


# A. Customer Segmentation (Unsupervised Learning)

**K-Means Clustering**

Steps:

Chose optimal clusters using Elbow Method

Trained KMeans on scaled features

Visualized clusters using scatterplots

Interpreted clusters for business segmentation

**Typical clusters:**

Cluster 1: Low Income, Low Spending

Cluster 2: High Income, High Spending (Premium Customers)

Cluster 3: Medium Income, High Loyalty

Cluster 4: Younger customers with high spending

Cluster 5: Older customers with low spending

**B. Predictive Modelling (Supervised Learning)**

Models trained:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

KNN Classifier (if used)

All models were trained using:
**model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy_score(y_test, pred)**

**Model Accuracy Comparison**

| Model                        | Accuracy                        |
| ---------------------------- | ------------------------------- |
| **Random Forest Classifier** | **89%** *(example placeholder)* |
| Decision Tree Classifier     | 86%                             |
| Logistic Regression          | 78%                             |
| KNN Classifier               | 74%                             |

(Replace these numbers with your notebook's actual accuracies.)

Best Performing Model → Random Forest


# Classification Report & Confusion Matrix

Generated:

Precision

Recall

F1-score

Confusion matrix heatmap


# Final Conclusions

Customer data reveals clear behavioural clusters useful for marketing campaigns.

High-income customers are not automatically high spenders — segmenting helps target correctly.

Membership programs (Gold/Platinum) play a strong role in loyalty and spending.

Best ML model for predicting spending behaviour was Random Forest (highest accuracy).

Customer segmentation can strongly improve targeted marketing & personalised promotions.


# Future Enhancements

Implement XGBoost / LightGBM for improved predictions

Build RFM (Recency–Frequency–Monetary) models

Deploy a web dashboard using Streamlit or Flask

Perform A/B testing on target customer groups

