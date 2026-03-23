import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
df=pd.read_csv(r"decision-tree-ml\\data\\train_u6lujuX_CVtuZ9i (1).csv")
#print(df.head())
#print(df.info())

# find missing values
#print(df.isnull().sum())

# Fill missing values
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Credit_History']=df['Credit_History'].fillna(1)

#df.fillna({'LoanAmount': df['LoanAmount'].mean(),'Credit_History': 1}, inplace=True)

#droping other null values
df.dropna(inplace=True)

df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})
#feature selection

X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

#train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Train Model
model = DecisionTreeClassifier(max_depth=5,criterion='gini')
model.fit(X_train, y_train)

#prediction
pred = model.predict(X_test)

#evalution
accuracy = accuracy_score(y_test, pred)
print("\nModel Accuracy:", accuracy)

#-------------------------------------------------------------------------------------------

# Using GridSearchCV for Hyperparameter Tuning
# -----------------------------
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10],
}

grid=GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,cv=5,scoring='f1')#scoring can have accuracy,recall,precision

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Best Model
best_model = grid.best_estimator_

#prediction
pred1=best_model.predict(X_test)
# Evaluation

accuracy = accuracy_score(y_test, pred1)
print("Test Accuracy:", accuracy)

#  Visualize Decision Tree
plt.figure(figsize=(20,15))

plt.subplot(1,2,1)

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True
)
plt.title("1 Decision Tree Visualization")

plt.subplot(1,2,2)

plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True
)
plt.title("2 Decision Tree  Visualization")
plt.savefig("decision-tree-ml/results/model2_plot.png")
plt.tight_layout()
plt.show()



