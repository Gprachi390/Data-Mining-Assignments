#!/usr/bin/env python
# coding: utf-8

# In[106]:


#importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import rand_score
from scipy.stats import mode
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


#load the datset
df = pd.read_csv("covtype_train.csv")


# In[3]:


df


# In[4]:


#checking the dataframe
df.info()


# In[5]:


#checking the target labels
df.groupby('target').size()


# In[6]:


df.shape


# In[7]:


#check if any null values
df.isnull().sum()


# In[8]:


#drop duplicate values in dataframe
df.drop_duplicates(inplace=True)
df


# In[9]:


#separating target column to use clustering algorithms
xx = df.drop(['target'], axis=1)
y = df['target']


# In[10]:


data = pd.get_dummies(xx)
data.head()


# In[11]:


#dimensionality reduction using tsne into two components tsne1 and tsne2
tsne = TSNE(n_components=2)
x_emb = tsne.fit_transform(data)


# In[12]:


x_emb


# In[13]:


data


# # Q1  (1) Representative object of each cluster

# #  K-means Clustering

# In[14]:


#compute k-means clustering
km = KMeans(n_clusters=7)
km.fit_predict(data)
y_km = km.labels_


# In[15]:


#attribute1: coordinate of cluster centres
print(km.cluster_centers_)


# In[16]:


#attribute2: cluster label of each point
print(km.labels_)


# # Hierarchical Clustering: Agglomerative Clustering

# In[17]:


#compute agglomerative clustering
am = AgglomerativeClustering(n_clusters=7,compute_distances=True)
am.fit_predict(data)
y_am = am.labels_


# In[18]:


#attribute1: cluster label of each point
print(am.labels_)


# In[19]:


#attribute2: distance between nodes in corresponding place in children(of non-leaf node)
print(am.distances_)


# In[20]:


#attribute3: number of leaves in hierarchical tree 
print(am.n_leaves_)
#children of each non-leaf node
am.children_


# # BIRCH Clustering

# In[21]:


#compute BIRCH clustering
br = Birch(n_clusters=7)
br.fit_predict(data)
y_br = br.labels_


# In[22]:


##attribute1: cluster label of each point
print(br.labels_)


# In[23]:


#attribute2: centroid of all subclusters
print(br.subcluster_centers_)


# # Gaussian Mixture Clustering

# In[66]:


#compute Gaussian mixture clustering (Gaussian Model)
gmm = GaussianMixture(n_components = 7)
gmm.fit(data)
y_gmm = gmm.predict(data)


# In[25]:


#attribute1: mean of each mixture component
print(gmm.means_)


# In[26]:


#attribute2: weights of each mixture component
print(gmm.weights_)


# # Q1 (2) Visualization of clusters

# In[27]:


#for K-means clustering
df1 = pd.DataFrame()
df1['tsne-1'] = x_emb[:,0]
df1['tsne-2'] = x_emb[:,1]
df1['Cluster'] = y_km
sns.FacetGrid(df1, hue="Cluster", height=6).map(plt.scatter, "tsne-1","tsne-2").add_legend()


# In[28]:


#for agglomerative clustering
df2 = pd.DataFrame()
df2['tsne-1'] = x_emb[:,0]
df2['tsne-2'] = x_emb[:,1]
df2['Cluster'] = y_am
sns.FacetGrid(df2, hue="Cluster",height=6).map(plt.scatter, "tsne-1","tsne-2").add_legend()


# In[29]:


#for BIRCH clustering
df3 = pd.DataFrame()
df3['tsne-1'] = x_emb[:,0]
df3['tsne-2'] = x_emb[:,1]
df3['Cluster'] = y_br
sns.FacetGrid(df3, hue="Cluster", height=6).map(plt.scatter, "tsne-1","tsne-2").add_legend()


# In[30]:


df4 = pd.DataFrame()
df4['tsne-1'] = x_emb[:,0]
df4['tsne-2'] = x_emb[:,1]
df4['Cluster'] = y_gmm
sns.FacetGrid(df4, hue="Cluster", height=6).map(plt.scatter, "tsne-1","tsne-2").add_legend()


# # Q1 (3) Cluster distribution with true label count

# In[77]:


#assign weights to each label of y(target)
def weight(y):
    length=len(y)
    weights={}
    for i in range(1,8):
        weights[i]= (length/(7*y.value_counts()[i]))%7
        
    return weights


# In[78]:


#print the calculated weights
weights=weight(y)
weights


# In[79]:


#function to check labels in y
def check_label(y):
    count=[0 for _ in range(8)]
    for i in range(1,8):
        try:
            c=np.bincount(y)[i]
        except:
            c=0
        count[i]=c*weights[i]
    return np.argmax(count)


# In[80]:


#function for true label count
check_y=check_label(y)
def tl_count(check_y,y):
    for i in range(7):
        print(f"Cluster:{i}")
        add=np.where(check_y==i)[0].tolist()
        add_y=y.iloc[add]
        print(add_y.value_counts())


# In[81]:


#value count of each label present in target column
y.value_counts()


# ### K-means clustering distribution with true label count 

# In[82]:


#K-means clustering distribution with true label count
tl_count(y_km,y)


# In[83]:


#check the count of datapoints present in each cluster: K-means
dataframe1 = pd.DataFrame()
dataframe1['kmean'] = km.labels_
dataframe1['kmean'].value_counts()


# ### Agglomerative clustering distribution with true label count

# In[84]:


#Agglomerative clustering distribution with true label count
tl_count(y_am,y)


# In[85]:


#check the count of datapoints present in each cluster: Agglomerative
dataframe2 = pd.DataFrame()
dataframe2['agglomerative'] = am.labels_
dataframe2['agglomerative'].value_counts()


# ### BIRCH clustering distribution with true label count

# In[86]:


#BIRCH clustering distribution with true label count
tl_count(y_br,y)


# In[87]:


#check the count of datapoints present in each cluster: Birch
dataframe3 = pd.DataFrame()
dataframe3['BIRCH'] = br.labels_
dataframe3['BIRCH'].value_counts()


# ### Gaussian mixture clustering distribution with true label count

# In[88]:


#Gaussian mixture clustering distribution with true label count
tl_count(y_gmm,y)


# In[89]:


#check the count of datapoints present in each cluster: Gaussian Mixture 
dataframe4 = pd.DataFrame()
dataframe4['GMM'] = y_gmm
dataframe4['GMM'].value_counts()


# # Q1 (4) Comaparision of cluster formation

# In[90]:


#rand score for K-means
rand_score(y,km.labels_)


# In[91]:


#rand score for agglomerative
rand_score(y,am.labels_)


# In[92]:


#rand score for BIRCH
rand_score(y,br.labels_)


# In[93]:


#rand score for gaussian mixture
rand_score(y,y_gmm)


# # Q2 Prediction of clusters

# In[94]:


def map_cl(y_target,y_label):
    cluster_map={}
    for i in range(7):
        add=np.where(y_label==i)[0].tolist()
        y=y_target.iloc[add]
        check_y=check_label(y)
        cluster_map[i]=check_y
    return cluster_map


# ### Checking which clustering model is giving better mapping
# ### Gaussian model and K-means are mapping the true labels more accurately

# In[95]:


#checking after mapping with GMM
cluster_map=map_cl(y,y_gmm)
cluster_map


# In[96]:


#function to assign clusters
def assign(cluster_map,y_label):
    yp=np.zeros(y_label.shape)
    for i in cluster_map:
        add=np.where(y_label==i)[0].tolist()
        yp[add]=cluster_map[i]
    return yp


# In[97]:


yp=assign(cluster_map,y_gmm)
f1_score(y,yp,average='weighted')


# In[52]:


#checking after mapping with K-means
cluster_map=map_cl(y,y_km)
cluster_map


# In[53]:


yp=assign(cluster_map,y_km)
f1_score(y,yp,average='weighted')


# In[54]:


#checking after mapping with Agglomerative clustering
cluster_map=map_cl(y,y_am)
cluster_map


# In[55]:


yp=assign(cluster_map,y_am)
f1_score(y,yp,average='weighted')


# In[56]:


#checking after mapping with BIRCH
cluster_map=map_cl(y,y_br)
cluster_map


# In[57]:


yp=assign(cluster_map,y_br)
f1_score(y,yp,average='weighted')


# ### Gaussian model is giving better score

# In[58]:


x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.20, random_state=42,stratify=y)


# In[59]:


gmm = GaussianMixture(n_components=7)
gmm.fit(x_train)
y_gmm=gmm.predict(x_train)
gmm.cluster_map=map_cl(y_train,y_gmm)


# In[ ]:


gmm.cluster_map=map_cl(y,y_gmm)
pickle.dump(gmm, open('model1', 'wb'))


# ### Checking the performance of the model on the data

# In[99]:


def predict(test_set) :
    prediction=[]
    model= pickle.load(open('model1','rb'))
    prediction=model.predict(test_set)
    yp=assign(model.cluster_map,np.array(prediction))
    return yp


# In[100]:


y_test.value_counts()


# In[101]:


yp=predict(x_test)


# In[102]:


yp1=pd.DataFrame(yp)
yp1.value_counts()


# In[103]:


f1_score(y_test,yp,average='weighted')


# In[105]:


print(classification_report(y_test,yp,zero_division=0))


# # PREDICT FUNCTION

# In[ ]:


#importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import rand_score
from scipy.stats import mode
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[107]:


def assign(cluster_map,y_label):
    yp=np.zeros(y_label.shape)
    for i in cluster_map:
        add=np.where(y_label==i)[0].tolist()
        yp[add]=cluster_map[i]
    return yp


# In[108]:


def predict(test_set) :
    prediction=[]
    model= pickle.load(open('final_model','rb'))
    testpd=pd.read_csv(test_set)
    testpd=pd.get_dummies(testpd)
    prediction=model.predict(testpd)
    yp=assign(model.cluster_map,np.array(prediction))
    return yp


# In[ ]:


#predict('test.csv')

