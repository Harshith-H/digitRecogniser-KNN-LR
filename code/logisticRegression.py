from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


data=fetch_openml('mnist_784')

X=data.data
y=data.target
y=np.array([int(x) for x in y])

X=pd.DataFrame(X)
y=pd.DataFrame(y)

a=np.ones(shape=[70000,1])
a=pd.DataFrame(a*5)
y=(y==a)


X_train, X_test, y_train, y_test = X[30000:37000], X[37000:40000], y[30000:37000], y[37000:40000]
X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)

sc=MinMaxScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)


lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print('Test Accuracy=', end=" ")
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print()


#Test Accuracy Obtained= 0.962
