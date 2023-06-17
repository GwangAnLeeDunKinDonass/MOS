#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def twod_visualization(df):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize=20)
    
    data = df.iloc[:,-1]
    targets = data.unique()
    for target in targets:
        indicesToKeep = (data == target)
        ax.scatter(df.loc[indicesToKeep, 'PC1']
                   , df.loc[indicesToKeep, 'PC2']
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    
def threed_visualization(df):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('3 component PCA', fontsize=20)
    
    data = df.iloc[:,-1]
    targets = data.unique()
    for target in targets:
        indicesToKeep = (data == target)
        ax.scatter(df.loc[indicesToKeep, 'PC1']
                   , df.loc[indicesToKeep, 'PC2']
                   , df.loc[indicesToKeep, 'PC3']
                   , s = 50)
    ax.legend(targets)
    ax.grid()

