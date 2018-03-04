import numpy as np
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("houseprices.csv")

X = data["area"]
Y = data["sale.price"]

X = np.array(X).reshape(len(X), 1)
Y = np.array(Y).reshape(len(Y), 1)

linearregression = lr()
linearregression.fit(X,Y)

predictX = linearregression.predict(X)

m = linearregression.coef_
b = linearregression.intercept_

a = np.arange(2000)

plt.scatter(X,Y)
plt.scatter(a,m*a+b, c="red")
plt.plot(X, predictX, color="blue")
plt.title("Linear Regression")
plt.show()
print("Mean Squared Error : ",mean_squared_error(X, Y))
print("Coefficient (m) : ", m)
print("Intercept (b) : ", b)