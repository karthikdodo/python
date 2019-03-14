from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('sample_stocks.csv')
#train,test=train_test_split(data,test_size=0.2)
Sum_of_Squarred_err=[]
K=range(1,15)
for k in K:
    clf = KMeans(n_clusters=k)
    clf.fit(data)
    Sum_of_Squarred_err.append(clf.inertia_)
plt.plot(K,Sum_of_Squarred_err,'bx-')
plt.xlabel("K values")
plt.ylabel("sum_of_squared_error")
plt.show()

#prediction=clf.predict(test)
#print(prediction)