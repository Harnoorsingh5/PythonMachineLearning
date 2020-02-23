import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate,train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

df = quandl.get('WIKI/GOOGL')
#print(df)

df = df [['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = math.ceil(0.01*len(df)) # length of df = 3424 -> 3424 * 0.01 = 34.24 -> ceil of 34.24 = 35
#forecast_out is used to calculate a value that he wants to see in future.
df['label'] = df[forecast_col].shift(-forecast_out) # shift forecast_col up by 35 places up and store it as new column 'label'
#shift function shifts specific DF column either up(-) or down(+).

X = np.array(df.drop(['label'],1)) # df.drop return a new data frame - and np.array converts this dataframe into an array of features
X = preprocessing.scale(X)# normalize the values # scale it along other values
X = X[:-forecast_out] # X from starting to last 35th element
print(len(X))
X_lately = X[-forecast_out:] # X from starting 35th element to last
print(len(X_lately))
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# can be commented as we are using pickle file for stored classifier
#clf = LinearRegression()
#clf = LinearRegression(n_jobs=10) #will run 10 jobs parellely for significantly fast training.
#clf.fit(X_train,y_train)
## it's good to store the trained model 
#with open('linearregression.pickle','wb') as f:
#    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test,y_test)
#print(accuracy) #accuracy for 35 days in advance

# clf2 = svm.SVR()
# clf2.fit(X_train,y_train)
# accuracy2 = clf2.score(X_test,y_test)
# print(accuracy2)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

style.use('ggplot')
df['Forecast'] = np.nan # adding new column in our df and setting all values to NaN

# we want to predict forecasts as tomorrow
# we have got amost 10 % of data (35 values) to forecast
# first we need to grad the last day in the data frame and begin assigning each new forecast a new day
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 # one day has 86400 seconds
next_unix = last_unix + one_day # we got the next day timestamp

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    #print(df.loc[next_date] ) 
## this data frame had date as the row index value
## we come in to the data frame row with the next date and put the predicted values in the last column i.e. Forecast column


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#Pickling is just serialization of python objects


#-------------------------------------------------
