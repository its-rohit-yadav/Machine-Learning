import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#loading dataset
df=pd.read_csv("multiple-linear-regression/data/Advertising.csv")

#printing first 5 records
print(df.head())

# Independent variable
X = df.iloc[:,1:4]

#target variable

y = df.iloc[:,-1]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#creating own linear regressor class :

class minelr:
  def __init__(self):
    self.coef=None
    self.intercept=None
  
  def fit(self,X_train,Y_train):

    #insert 1
    X_train=np.insert(X_train,0,1,axis=1)

    result=np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(Y_train)

    self.coef=result[1:]

    self.intercept=result[0]

    return self.coef,self.intercept
  
  #prediction 
  def predict(self,X_test):
      return np.dot(X_test,self.coef)+self.intercept

# Create model through Minelr
model1 = minelr()

# Train model
model1.fit(X_train, y_train)

# Prediction through MineLr
prediction = model1.predict(X_test)

# Model performance
r2 = r2_score(y_test, prediction)

#Model performance
r2_own = r2_score(y_test, prediction)


# Create model
model2 = LinearRegression()

# Train model
model2.fit(X_train, y_train)

# Prediction
prediction2 = model2.predict(X_test)

# Model performance
r2_sklearn = r2_score(y_test, prediction2)

print("Through sklearn linear regression R2 Score:", r2_sklearn)
print("Through own minelr R2 Score:", r2_own)


plt.figure(figsize=(12,5))

# Graph 1 : MineLR predictions
line = [y_test.min(), y_test.max()]
plt.subplot(1,2,1)
plt.scatter(y_test, prediction, color='blue')
plt.plot(line,line,color='red', linestyle='--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("MineLR Model")

# Graph 2 : Sklearn Linear Regression
plt.subplot(1,2,2)
plt.scatter(y_test, prediction2, color='green')
plt.plot(line,line,color='red', linestyle='--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Sklearn LinearRegression")
plt.savefig("multiple-linear-regression/results/model_plot.png")
plt.tight_layout()
plt.show()