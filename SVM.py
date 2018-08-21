import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
from matplotlib import pyplot
from scipy import spatial

df = pd.read_csv("Datasets/breast_cancer_data.txt")
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'],1))
Y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)

prediction = clf.predict(X_test)

print prediction
print Y_test
print "Similarity Score:",(1-spatial.distance.cosine(Y_test,prediction))*100

pyplot.scatter(range(len(Y_test)),Y_test,c='g')
pyplot.scatter(range(len(prediction)),prediction,c='b')
#pyplot.show()
