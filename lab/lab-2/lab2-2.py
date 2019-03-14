from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
data=pd.read_csv('sample_stocks.csv')
data = data.dropna(how='any',axis=0)
#train,test=train_test_split(data,test_size=0.2)
clf = KMeans(n_clusters=3)
answer=clf.fit_predict(data)
score=silhouette_score(data,answer)
print(score)
color_theme=np.array(['darkgray','lightsalmon','powderblue'])

plt.scatter(x=data['returns'],y=data['dividendyield'],c=color_theme[clf.labels_],s=50)
plt.show()
#print(prediction)