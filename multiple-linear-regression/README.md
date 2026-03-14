# Simple Linear Regression 

This repository contains my implementation of **Simple Linear Regression built from scratch using Python and NumPy**.
The goal of this project was to understand how linear regression works internally instead of relying directly on machine learning libraries.

The model is implemented using the **Normal Equation**, which calculates the optimal parameters mathematically without using iterative optimization.

## Mathematical Idea :

The regression coefficients are computed using:

β = (XᵀX)⁻¹Xᵀy

This equation directly gives the best fitting parameters for the linear model.

## What this project includes :

* A custom Linear Regression class
* Implementation of `fit()` and `predict()` methods
* Manual handling of the bias (intercept) term
* Matrix operations using NumPy
* Example usage with a dataset

## Example Usage

```python
model = MineLR()

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
```

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
