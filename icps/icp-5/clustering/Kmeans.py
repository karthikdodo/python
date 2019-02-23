from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('/Users/karthikchowdary/Desktop/KarthiK/graduate/spring-19/python/class-5/Python_Lesson5/data/Iris.csv'  )
data = pd.read_csv('/Users/karthikchowdary/Desktop/KarthiK/graduate/spring-19/python/class-5/Python_Lesson5/data/College.csv'  )
x = dataset.iloc[:,[1,2,3,4]]
y = dataset.iloc[:-1]
# see how many samples we have of each species
print(dataset["Species"].value_counts())

sns.FacetGrid(dataset, hue="Species", size=4).map(plt.scatter, "SepalLengthCm", "SepalWidthCm")
# do same for petals
sns.FacetGrid(dataset, hue="Species", size=4).map(plt.scatter, "PetalLengthCm", "PetalWidthCm")
plt.show()
# note that the species are nearly linearly separable with petal size,but sepal sizes are more mixed.
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
x1=data.iloc[:,[4,17]]
"""X_array = np.array(data['Accept'])
Y_array=np.array(data['Expend'])"""
print(x1)

normalized_X=preprocessing.normalize([x1])
print(normalized_X)

from sklearn.cluster import KMeans

nclusters = 3 # this is the k in kmeans
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(normalized_X)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(normalized_X)
print(y_cluster_kmeans)

"""from sklearn import metrics
wcss = []
##elbow method to know the number of clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()"""
