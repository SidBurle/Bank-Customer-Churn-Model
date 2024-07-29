# Bank-Customer-Churn-Model
Project Overview
The Bank Customer Churn Model is designed to predict the likelihood of customers leaving a bank, a phenomenon known as customer churn. This project leverages various machine learning techniques and data preprocessing methods to build a robust predictive model. By identifying potential churners, banks can take proactive measures to retain valuable customers and minimize revenue loss.

Objective
The main objectives of the project are as follows:

Data Scaling: Standardizing the data to ensure uniformity and improve model performance.
Feature Scaling: Applying scaling techniques to numerical features to balance their impact on the model.
Handling Imbalanced Data: Addressing the issue of class imbalance in the dataset using:
Random Under Sampling: Reducing the number of majority class instances to match the minority class.
Random Over Sampling: Increasing the number of minority class instances to balance the classes.
Support Vector Machine (SVM) Classifier: Utilizing SVM, a powerful classifier, to predict customer churn.
Grid Search for Hyperparameter Tuning: Optimizing model performance by tuning hyperparameters through Grid Search.
Data Source
The dataset used in this project is sourced from the YBI Foundation Dataset repository. The dataset contains various customer attributes and labels indicating whether a customer churned or not.

Technologies and Libraries Used
Python: Core programming language for data analysis and machine learning.
Pandas: Data manipulation and analysis.
NumPy: Numerical computing.
Matplotlib & Seaborn: Data visualization.
Scikit-learn: Machine learning library, including tools for model building, evaluation, and data preprocessing.
Imbalanced-learn: Handling imbalanced datasets.
Data Import and Preprocessing
Data Import: The dataset is imported using pandas.

python
Copy code
import pandas as pd
dt = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Bank%20Churn%20Modelling.csv')
Data Exploration: Basic data exploration, including info, description, and checking for duplicates.

python
Copy code
dt.info()
dt.describe()
dt.duplicated('CustomerId').sum()
Data Cleaning: Handling missing values, encoding categorical variables, and dropping unnecessary columns.

Geography and Gender columns are encoded with numerical values.
Handling customers with zero balance by creating a new feature 'Zero Balance'.
python
Copy code
dt.replace({'Geography': {'France': 2, 'Germany': 1, 'Spain': 0}}, inplace=True)
dt.replace({'Gender': {'Male': 0, 'Female': 1}}, inplace=True)
dt['Zero Balance'] = np.where(dt['Balance'] > 0, 1, 0)
Defining Target and Features: Splitting the dataset into features (X) and target (y).

python
Copy code
X = dt.drop(['Surname', 'Churn'], axis=1)
y = dt['Churn']
Handling Imbalanced Data
Random Under Sampling (RUS): Balancing the dataset by reducing the number of majority class samples.

python
Copy code
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=2529)
X_rus, y_rus = rus.fit_resample(X, y)
Random Over Sampling (ROS): Increasing the number of minority class samples.

python
Copy code
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=2529)
X_ros, y_ros = ros.fit_resample(X, y)
Data Scaling
Standardization of numerical features to ensure consistency.

python
Copy code
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']
X_train[features] = sc.fit_transform(X_train[features])
X_test[features] = sc.transform(X_test[features])
Model Building and Evaluation
Support Vector Machine Classifier: Initial model building using SVM.

python
Copy code
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
Evaluation: Model evaluation using confusion matrix and classification report.

python
Copy code
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
Grid Search for Hyperparameter Tuning: Optimizing the SVM model using Grid Search.

python
Copy code
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf'], 'class_weight': ['balanced']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=2)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
Model Results
Confusion matrix and classification report are generated for each model (original, under-sampled, over-sampled).
The best model parameters and performance metrics are recorded.
Conclusion
A Bank Customer Churn Model is an essential tool for banks to predict customer churn. By understanding the patterns and factors leading to churn, banks can implement targeted retention strategies. This model demonstrates the application of various data preprocessing techniques, handling class imbalance, and utilizing SVM for classification, followed by hyperparameter optimization using Grid Search.

