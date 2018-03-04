import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures as pf
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("SalaryPosition.csv")

X = data["Salary"]
Y = data["Level"]

X = np.array(X).reshape(len(X), 1)
Y = np.array(Y).reshape(len(Y), 1)

lregression = lr()
lregression.fit(X,Y)

lregression2 = pf()
X_Pol = lregression2.fit_transform(X)

predictX = lregression.predict(X)

m = lregression.coef_
b = lregression.intercept_

a = np.arange(12000)

plt.scatter(X,Y)
plt.plot(X, predictX, color="blue")
plt.scatter(a, a*m+b, color="red")
plt.title("Salary Position")


print(mean_squared_error(X,Y))
plt.show()