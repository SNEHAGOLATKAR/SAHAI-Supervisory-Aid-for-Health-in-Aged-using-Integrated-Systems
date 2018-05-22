# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:11:01 2017

@author: SNEHA
"""
# More about svm reffer this link
# https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import svm
#from matplotlib import style
#style.use("ggplot")
# import some data to play with
import pandas as pd
data = pd.read_csv(r'C:\Users\SNEHA\Desktop\Final_year\y_cor_binary.csv')
test = pd.read_csv(r'C:\Users\SNEHA\Desktop\Final_year\y_cor5.csv')
test7 = pd.read_csv(r'C:\Users\SNEHA\Desktop\Final_year\data7.csv')

# shape of dataset
print("Shape:", data.shape)
 
# column names
print("\nFeatures:", data.columns)
 
# storing the feature matrix (X) and response vector (y)
X = data[data.columns[:-1]]
y = data[data.columns[-1]] 

# printing first 5 rows of feature matrix
print("\nFeature matrix:\n", X.head())
 
# printing first 5 values of response vector
print("\nResponse vector:\n", y.head())


# splitting X and y into training and testing sets
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=1)

#training x and y data using SVM with linear kernel
svc = svm.SVC(kernel='linear', C=1,gamma=10).fit(X, y)

#Testing trained models with different feature values
print(svc.predict([[1,1028.07,748.96,748.117,749.803,514.242,498.464,299.065,277.594,531.605,313.699,314.802,-129.27,-115.699,-492.418,-417.946],
                   [1,-1028.07,748.96,748.117,749.803,514.242,498.464,299.065,277.594,531.605,313.699,314.802,-129.27,-115.699,-492.418,-417.946],
                   [1,-1028.07,-748.96,748.117,749.803,514.242,498.464,299.065,277.594,531.605,313.699,314.802,-129.27,-115.699,-492.418,-417.946],
                   [1,-1028.07,-748.96,-748.117,749.803,514.242,498.464,299.065,277.594,531.605,313.699,314.802,-129.27,-115.699,-492.418,-417.946],
                   [1,-1028.07,-748.96,-748.117,-749.803,514.242,498.464,299.065,277.594,531.605,313.699,314.802,-129.27,-115.699,-492.418,-417.946],
                   [1,-1028.07,-748.96,-748.117,-749.803,-514.242,-498.464,-299.065,-277.594,-531.605,-313.699,-314.802,-129.27,-115.699,-492.418,-417.946]]))

"""    print(svc.predict([[1,	1028.07,	748.96,	748.117,	749.803,	514.242,	498.464,	299.065,	277.594,	531.605,	313.699,	314.802,	-129.27,	-115.699,	-492.418,	-417.946
],[1,	1028.07,	748.96,	748.117,	749.803,	514.242,	498.464,	299.065,	277.594,	531.605,	313.699,	314.802,	-129.27,	-115.699,	-492.418,	-417.946
],[1,	-506.922,	573.552,	688.111,	458.993,	-628.121,	-386.283,	-670.246,	-418.049,	-662.13,	-867.415,	-634.002,	-1241.88,	-832.966,	-1674.66,	-1241.34
],[1,	-518.139,	-432.658,	-443.8,	-421.516,	522.904,	-513.061,	-751.462,	-479.702,	-583.517,	-739.777,	-728.974,	-1114.24,	-927.938,	-1547.02,	-1336.31],
[1,	980.742,	693.347,	705.582,	681.112,	473.583,	438.4,	154.623,	184.343,	485.391,	284.262,	270.607,	-174.804,	-178.69,	-569.08,	-559.371]]))
"""
#test.User[85]
#predicting the model by other datasets
predictions = svc.predict(test)
#predictions = svc.predict(test7)
print(predictions)
c=0
for n,row in enumerate(predictions) :
    if row == 1 :
        print(test.User[n])
    #if predictions [n] == 1:
       # print(test[n])
#accuracy = accuracy_score(predictions)
#print( accuracy)
#from sklearn import metrics
##print("kNN model accuracy:", metrics.accuracy_score(test7, predictions))
#for i,r in enumerate(predictions):
 #   print (i)
    #print (r)
"""    w = svc.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - svc.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter (X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()  """



















