#!/usr/bin/env python
# coding: utf-8

# In[2]:


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



def assign(cluster_map,y_label):
    yp=np.zeros(y_label.shape)
    for i in cluster_map:
        add=np.where(y_label==i)[0].tolist()
        yp[add]=cluster_map[i]
    return yp


def predict(test_set) :
    prediction=[]
    model= pickle.load(open('final_model','rb'))
    testpd=pd.read_csv(test_set)
    testpd=pd.get_dummies(testpd)
    prediction=model.predict(testpd)
    yp=assign(model.cluster_map,np.array(prediction))
    return yp


# In[3]:


#predict('test.csv')

