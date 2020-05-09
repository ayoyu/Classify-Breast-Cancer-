# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:23:52 2018

@author: Ayoub El khallioui
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from urllib.request import urlopen
url_csv='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'

def dowload(url_csv):
    response=urlopen(url_csv)
    csv=response.read()
    csv_str=str(csv)
    return csv_str

clean=dowload(url_csv).split('\\n')
clean1=clean[:-1]
clean2=[]
for d in clean1:
    clean2.append(d.split(','))

data=pd.DataFrame(clean2,columns=['id number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','target'])
r=data['id number'][0].split("b'")
data['id number'][0]=data['id number'][0].replace(data['id number'][0],r[1])
def handel_missing_values(dataset, missing_values_header, missing_label):
    return dataset[dataset[missing_values_header] != missing_label]
headers=['id number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','target']      

data1=handel_missing_values(data,headers[6],'?')
data2=handel_missing_values(data1,headers[7],'?')
data3=handel_missing_values(data2,headers[8],'?')
data4=handel_missing_values(data3,headers[9],'?')
data5=handel_missing_values(data4,headers[10],'?')

for i in headers:
    data5[i]=data5[i].apply(lambda x:int(x))

X=data5[headers[:-1]].values
y=data5[headers[-1]].values        

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
RF=RandomForestClassifier(n_estimators=40,criterion='gini',max_depth=20)        
model=RF.fit(X_train,y_train)    
y_pred=model.predict(X_test)    
accuracy_score(y_test,y_pred)
#accuracy_score(y_train,model.predict(X_train))    
print('Missclassification for our model is %d'%(y_pred!=y_test).sum())
conf=confusion_matrix(y_test,y_pred)
import seaborn as sns
sns.heatmap(conf, square=True, annot=True, cbar=False)


RF_entropy=RandomForestClassifier(n_estimators=40,criterion='entropy',max_depth=20)
y_predEn=RF_entropy.fit(X_train,y_train).predict(X_test)
accuracy_score(y_test,y_predEn)
print('Missclassification for our model is %d'%(y_predEn!=y_test).sum())

from sklearn.tree import export_graphviz
export_graphviz(RF.estimators_,feature_names=data5[headers[:-1]].columns,filled=True,rounded=True)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import _tree

def leaf_depths(tree, node_id = 0):
    
     '''
     tree.children_left and tree.children_right store ids
     of left and right chidren of a given node
     '''
     left_child = tree.children_left[node_id]
     right_child = tree.children_right[node_id]

     '''
     If a given node is terminal, 
     both left and right children are set to _tree.TREE_LEAF
     '''
     if left_child == _tree.TREE_LEAF:
         
         '''
         Set depth of terminal nodes to 0
         '''
         depths = np.array([0])

     else:
         
         '''
         Get depths of left and right children and
         increment them by 1
         '''
         left_depths = leaf_depths(tree, left_child) + 1
         right_depths = leaf_depths(tree, right_child) + 1
 
         depths = np.append(left_depths, right_depths)
 
     return depths
def leaf_samples(tree, node_id = 0):
    
     left_child = tree.children_left[node_id]
     right_child = tree.children_right[node_id]

     if left_child == _tree.TREE_LEAF:
        
         samples = np.array([tree.n_node_samples[node_id]])

     else:
        
         left_samples = leaf_samples(tree, left_child)
         right_samples = leaf_samples(tree, right_child)

         samples = np.append(left_samples, right_samples)

     return samples
def draw_ensemble(ensemble):

     plt.figure(figsize=(8,8))
     plt.subplot(211)

     depths_all = np.array([], dtype=int)

     for x in ensemble.estimators_:
         tree = x.tree_
         depths = leaf_depths(tree)
         depths_all = np.append(depths_all, depths)
         plt.hist(depths, histtype='step', color='#ddaaff', 
                  bins=range(min(depths), max(depths)+1))

     plt.hist(depths_all, histtype='step', color='#9933ff', 
              bins=range(min(depths_all), max(depths_all)+1), 
              weights=np.ones(len(depths_all))/len(ensemble.estimators_), 
              linewidth=2)
     plt.xlabel("Depth of leaf nodes")
    
     samples_all = np.array([], dtype=int)
    
     plt.subplot(212)
    
     for x in ensemble.estimators_:
         tree = x.tree_
         samples = leaf_samples(tree)
         samples_all = np.append(samples_all, samples)
         plt.hist(samples, histtype='step', color='#aaddff', 
                  bins=range(min(samples), max(samples)+1))
    
     plt.hist(samples_all, histtype='step', color='#3399ff', 
              bins=range(min(samples_all), max(samples_all)+1), 
              weights=np.ones(len(samples_all))/len(ensemble.estimators_), 
              linewidth=2)
     plt.xlabel("Number of samples in leaf nodes")
    
     plt.show()
draw_ensemble(RF)
"""
from sklearn import tree
i_tree = 0
for tree_in_forest in RF.estimators_:
    with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
        my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
    i_tree = i_tree + 1

from sklearn.tree import export_graphviz
export_graphviz(RF,feature_names=data5[headers[:-1]].columns,filled=True,rounded=True)

"""

