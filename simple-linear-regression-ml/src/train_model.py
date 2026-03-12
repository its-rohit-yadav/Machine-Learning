import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("data/Salary_dataset.csv")

# Independent variable
X = df.iloc[:,1:2]

# Dependent variable
y = df.iloc[:,-1]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Model performance
r2 = r2_score(y_test, prediction)
print("R2 Score:", r2)

actual_salary = y_test.iloc[0]

predicted_salary = model.predict(X_test.iloc[[0]])

print("Experience:", X_test.iloc[0].values[0])
print("Actual Salary:", actual_salary)
print("Predicted Salary:", predicted_salary[0])
# Plot regression line
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience Linear Regression")
plt.savefig("results/regression_plot.png")
plt.show()