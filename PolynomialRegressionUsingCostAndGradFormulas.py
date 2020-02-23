import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

myData = pd.read_csv('position_salaries.csv')
#print(myData)

X = myData.iloc[:, 1:2].values 
y = myData.iloc[:, 2:3].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
lin_reg = LinearRegression()
lin_reg.fit(X,y)

def viz_linear():
    plt.scatter(X,y,color='red')
    plt.plot(X,lin_reg.predict(X),color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return

viz_linear()

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly,y)

def viz_polynomial():
    plt.scatter(X,y,color='red')
    plt.plot(X,pol_reg.predict(X_poly),color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return 

viz_polynomial()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[5.5]]))
#output should be 249500

# Predicting a new result with Polymonial Regression
print(pol_reg.predict(poly_reg.fit_transform([[5.5]])))
#output should be 132148.43750003