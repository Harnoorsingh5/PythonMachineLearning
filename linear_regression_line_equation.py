from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)
 
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - (mean(xs * ys))) / ((mean(xs))**2 - mean(xs**2)) )
    b = ( mean(ys) - ( m * mean(xs) ) )
    return m,b

def squared_error(y, y_hat):
    return sum((y-y_hat)**2)

def coefficient_of_determination(y, y_hat):
    '''
        R-Squared
    '''
    y_mean_line = [mean(y) for _ in y ]
    squared_error_fit_line = squared_error(y, y_hat)
    squared_error_mean_line = squared_error(y, y_mean_line)
    return 1 - (squared_error_fit_line/squared_error_mean_line)
    
xs, ys = create_dataset(40, 10, 2, correlation='pos')

m,b = best_fit_slope(xs, ys)
print(m,b)

y_hat = [ m*x+b for x in xs ]


predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(ys, y_hat)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color = 'g')
plt.plot(xs,y_hat)
plt.show()
