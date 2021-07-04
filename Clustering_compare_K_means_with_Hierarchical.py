
"""The below code looks at how the K-means and Hierarchical clusters are calculated in Python"""


"""# Importing packages"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

"""# Import the penguin dataset"""
penguins = sns.load_dataset("penguins")

"""#Typical queries used to see what your data looks like - always carry this out before completing any analysis
    on your data"""
penguins.head()
penguins.info()
penguins.describe()
penguins.columns
penguins.isnull().sum()

"""# We are interested in bill length and bill depth for clustering purposes, while usually you would
     look at what is the best way of dealing with Null values, in this scenario we will simply remove
     Null values"""

penguins_red = penguins.dropna(subset=['bill_length_mm', 'bill_depth_mm'], how='any')

penguins_red.reset_index(inplace=True)



"""# Let us take a look at clustering bill_length_mm and bill_depth_mm together
   # Very important to visually see how many clusters it looks like the data could be split into
   # Looks likely to be 2-3 clusters"""
sns.set()
_ = sns.scatterplot(data=penguins, x='bill_length_mm', y='bill_depth_mm')
_ = plt.xlabel('Bill length (mm)')
_ = plt.ylabel('Bill depth (mm)')
_ = plt.title('Bill length v Bill depth', fontsize=20)
plt.plot()


"""#When you include species in the graph you can see that there is a clear split in bill length v bill depth 
    for each of the species"""
#plt.clf()
sns.set()
_ = sns.scatterplot(data=penguins, x='bill_length_mm', y='bill_depth_mm', hue='species')
_ = plt.xlabel('Bill length (mm)')
_ = plt.ylabel('Bill depth (mm)')
_ = plt.title('Bill length v Bill depth', fontsize=20)
plt.plot()


""" # Penguins reduced dataset including only bill length and depth"""
bill_dataset = penguins.iloc[:,2:4].values  #  2d array containing bill length and bill depth


##############################################################################################################
                                        # K means
##############################################################################################################

"""# Using the elbow method in K means clustering to find the optimal number of clusters 
   # WCSS is the sum of the squared distances from each point in a cluster to the centre of the cluster.
   # init refers to the initial cluster centres. k-means ++ speeds up convergence.
   # While 2 and 3 look like reasonable choices we will go with 3 but any of 2-4 look reasonable
   # (unfortunately choosing the number of clusters is an art rather than a science)"""

#plt.clf()
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)  # Firstly call the algorithm
    kmeans.fit(bill_dataset)  # fit is always used to train an algorithm
    wcss.append(kmeans.inertia_)  # inertia_ gives us the wcss value for each cluster.
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method',fontsize=20)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

"""# Training the K-Means model on the dataset"""
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 2)
bill_kmeans = kmeans.fit_predict(bill_dataset)
print(bill_kmeans)

##############################################################################################################
                                       # Hierarchical
##############################################################################################################

""" # Using the dendrogram to find the optimal number of clusters
    # Plot the dendrogram 
    # Ward consists of minimising the variance within your clusters
    # The number of clusters equals the number of lines you cross at the largest Euclidean distance resulting
    # from adding a cluster
    # Again 2,3 or 4 look like reasonable choices (2 offers the greatest increase in Euclidian distance but
    # we will go with 3 as we want to compare against K means approach)
    # The following KDnuggets page is extremely useful for understanding Hierarchical clustering 
    # https://www.kdnuggets.com/2019/09/hierarchical-clustering.html"""

plt.clf()
dendrogram = sch.dendrogram(sch.linkage(bill_dataset, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Bill info')
plt.ylabel('Euclidean distances')
plt.show()

"""# Training the Hierarchical Clustering model on the dataset
   # affinity = type of distance you want to look at.
   # linkage = 'ward' consists of minimising the variance within your clusters"""

hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
bill_hc_clusters = hc.fit_predict(bill_dataset)



##############################################################################################################
                                       # Check results
##############################################################################################################

"""# Include the clusters in our main dataset"""

penguins_cluster = pd.concat([penguins_red
                                 , pd.DataFrame(bill_kmeans,columns=['kmeans_clus'])
                                 ,pd.DataFrame(bill_hc_clusters, columns=['hierarch_clus'])], axis=1)

print(penguins_cluster)


""" # Plot bill length and depth again except this time split the data by both sets of clusters"""

# plt.clf()
_ = sns.scatterplot(data=penguins_cluster, x='bill_length_mm', y='bill_depth_mm', hue= 'kmeans_clus')
_ = plt.xlabel('Bill length (mm)')
_ = plt.ylabel('Bill depth (mm)')
_ = plt.title('Bill length v Bill depth', fontsize=20)
plt.plot()

_ = sns.scatterplot(data=penguins_cluster, x='bill_length_mm', y='bill_depth_mm', hue= 'hierarch_clus')
_ = plt.xlabel('Bill length (mm)')
_ = plt.ylabel('Bill depth (mm)')
_ = plt.title('Bill length v Bill depth', fontsize=20)
plt.plot()

"""# How close are the clusters to the species?
     Please note this is not the purpose of clustering!!"""

penguins_cluster.pivot_table(index=['species','kmeans_clus'], columns=['hierarch_clus']
                             , values='body_mass_g', aggfunc=len, fill_value=0)

"""As you can see clustering is not a technique for predicting a variable it is an approach from grouping
   your features. For anyone who has completed the Andrew Ng course on Machine Learning he describes clustering
   as more of an art than a science whereby you may have some prior reason or knowledge on the clusters e.g.
   you are looking to cluster sizes into small, medium and large in this case the above two approaches are 
   extremely useful"""
