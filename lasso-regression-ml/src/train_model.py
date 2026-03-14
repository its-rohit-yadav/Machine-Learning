#importing all libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error

# Load Dataset :
df=pd.read_csv('lasso-regression-ml/data/housing.csv')

print(df.info())

print(df.describe())

# Fill missing values
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# Convert categorical feature
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Split features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Scaling

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Lasso with different value
alphas=[0.0001, 0.001, 0.01, 0.1, 1,10]

r2_scores = []
mse_score = []

for alpha in alphas:
    model=Lasso(alpha=alpha)

    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    r2=r2_score(y_test,y_pred)

    mse = mean_squared_error(y_test,y_pred)

    r2_scores.append(r2)

    mse_score.append(mse)

    print("Alpha:", alpha)

    print("R2 Score:", r2)

    print("MSE:", mse)

# Graph Alpha vs R2

plt.figure(figsize=(10,6))

plt.plot(alphas, r2_scores, marker="o")

plt.xscale("log")

plt.xlabel("Alpha Value")

plt.ylabel("R2 Score")

plt.title("Effect of Alpha on Lasso Performance")

plt.savefig("lasso-regression-ml/results/model_plt.png")

plt.show()

 #find best aplha value 

best_score = max(r2_scores)

index = r2_scores.index(best_score)

best_alpha = alphas[index]

print("Best Alpha:", best_alpha)

#Graph best model

best_model = Lasso(alpha=best_alpha)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

plt.figure(figsize=(7,6))

plt.scatter(y_test, y_pred)

plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()],
         linestyle='--',
         color='red')

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")

plt.title(f"Prediction Plot (Best Alpha = {best_alpha})")

plt.savefig("lasso-regression-ml/results/bestmodel_plt.png")

plt.show()

# sample = X_test[0].reshape(1, -1)

# prediction = best_model.predict(sample)

# print("Predicted price:", prediction)
# print("Actual price:", y_test.iloc[0])

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(comparison.head(10))