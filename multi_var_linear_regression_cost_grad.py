import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#myData = np.genfromtxt('home.txt',delimiter = ',',dtype=np.int32) ## stores csv file into array
myData = pd.read_csv('home.txt',names=["size","bedroom","price"]) ## stores csv file into data frame
print(myData.head())

## As the data is not on same scale, it will be difficult ro run regression on this data set.
## So, in order to prevent this we need to normalize the data
## Normalize formula =  (value - mean) / standard deviation

myData = (myData - myData.mean()) / myData.std()
# print(myData.head())

#X = myData.loc[:,["size","bedroom"]]
X = myData.iloc[:,0:2]
ones = np.ones([X.shape[0],1]) ## creates a ones array of -> number of rows = number of rows in X and -> number of columns = 1
X = np.concatenate((ones,X),axis = 1) ##number of rows in X * 3
#print(X)

#y = myData.loc[:,["price"]]
y = myData.iloc[:,2:3].values ## .values converts myData dataframe to numpy array.
## y-> number of rows in X * 1
#print(y)

theta = np.zeros([1,3]) ## 1 * 3

alpha = 0.01
iters = 1000

def computeCost(X,y,theta):
    J = (1/(2*len(X))) * np.sum(np.power(((X @ theta.T) - y),2))  ## X*3 * 3*1 => X*1 => after sum of these => 1 => then take average
    return J

cost = computeCost(X,y,theta)
# print(cost)

def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - alpha * (1/len(X)) *  np.sum(((X @ theta.T) - y) * X, axis=0)
        cost[i] = computeCost(X,y,theta)

    return (theta,cost)

g,J = gradientDescent(X,y,theta,iters,alpha)
print(g,J)

fig, ax = plt.subplots()  
ax.plot(np.arange(iters), J, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')  
plt.show()