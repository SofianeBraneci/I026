# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 23:01:07 2020

@author: hp
"""
import numpy as np
from Classifiers import ClassifierKNN, ClassifierPerceptron, ClassifierPerceptronKernel, KernelPoly, ClassifierLineaireRandom
from utils import *
#np.random.seed(42)


def cross_validation(classifier, data, labels, m=4):
    """ 
        train on m - 1 fold, test on m-th fold 
        Cross Validation correctife
    
    """
    data_split = list()
    label_split = list()
    
    fold_size = len(data) // m
    print(fold_size)
    
    for _ in range(m):
        indices = np.random.randint(0, len(data), size=fold_size)
        data_split.append(data[indices])
        label_split.append(labels[indices])
        labels = np.delete(labels, indices)
        data = np.delete(data, indices, axis=0)
        print(len(data))
    
   # data to test upon
    test_data = None
    test_label = None
    train_acc = []
    test_acc = []
    for i in range(m):
       test_data = data_split[i]
       test_label = label_split[i]
       train_data = []
       train_labels = []

       # construct the training data
       for d, l in zip(data_split, label_split):
           if np.all(d != test_data) and np.all(l != test_label):
               train_data.append(d)
               train_labels.append(l)
        #-----------------------------
       train_data = np.vstack((train_data))
       train_labels = np.hstack((train_labels))
       classifier.train(train_data, train_labels)
       acc_train = classifier.accuracy(train_data, train_labels)
       acc_test = classifier.accuracy(test_data, test_label)
        
       test_acc.append(acc_test)
       train_acc.append(acc_train)
    print('train accuracy', train_acc)
    print('test accuracy', test_acc)
    return (np.mean(train_acc), np.std(train_acc)),(np.mean(test_acc), np.std(test_acc)) 

        
def cross_validation_loo(classifier, data, labels, k=10):
    """
        LOO : leave one out
    
    """

    size = len(data) // k
    data_copy = list(data)
    labels_copy = list(labels)
    data_indices = []
    labels_indices = []
    for _ in range(k):
        fold_data = []
        fold_labels = []
        for _ in range(size):
            index = np.random.randint(len(data_copy))
            data_copy.pop(index)
            labels_copy.pop(index)
            fold_data.append(index)
            fold_labels.append(index)
        data_indices.append(fold_data)
        labels_indices.append(fold_labels)
    train_acc = [0]
    test_acc = [0]
    for i in range(k):
       test_data = data[data_indices[i]]
       test_label = labels[labels_indices[i]]
       train_data = []
       train_labels = []
       for x, y in zip(data_indices, labels_indices):
           if np.all(x != data_indices[i]) and np.all(labels[y] != labels_indices[i]) :
               train_data.append(data[x])
               train_labels.append(labels[y])
       train_data = np.array(train_data)    
       train_labels = np.array(train_labels)
       
       train_data = np.vstack(train_data)
       train_labels = np.hstack(train_labels)
       
       classifier.train(train_data, train_labels)
       
       train_acc.append(classifier.accuracy(train_data, train_labels))
       test_acc.append(classifier.accuracy(test_data, test_label))
       
    return (np.mean(train_acc), np.std(train_acc)),(np.mean(test_acc), np.std(test_acc))    
##  
def cross_validation_l(classifiers, data, labels, k=10):
    """ cross validation of mutiple classifiers
        --------------
        classifiers: list of classifiers
        data: 2d array
        labels: 1d array 
        k: number of folds
        --------------
        return dict
    """
    
    
    size = len(data) // k
    data_copy = list(data)
    labels_copy = list(labels)
    data_indices = []
    labels_indices = []
    for _ in range(k):
        fold_data = []
        fold_labels = []
        for _ in range(size):
            index = np.random.randint(len(data_copy))
            data_copy.pop(index)
            labels_copy.pop(index)
            fold_data.append(index)
            fold_labels.append(index)
        data_indices.append(fold_data)
        labels_indices.append(fold_labels)
    train_acc = []
    test_acc = []
    d = {}
    name = ''
    names = []
    i = 1
    for c in classifiers:
        if c.__class__.__name__ in d.keys():
            name = c.__class__.__name__ + str(i)
            i += 1
            names.append(name)
        else:
            name = c.__class__.__name__
            names.append(name)
        d[name] = {'test_acc_list':[], 'train_acc_list':[],
                                 'train_acc':None,'train_std':None,'test_std':None, 'test_acc':None }
    
    for i in range(k):
       test_data = data[data_indices[i]]
       test_label = labels[labels_indices[i]]
       train_data = []
       train_labels = []
       for x, y in zip(data_indices, labels_indices):
           if np.all(x != data_indices[i]) and np.all(labels[y] != labels_indices[i]) :
               train_data.append(data[x])
               train_labels.append(labels[y])
       train_data = np.array(train_data)    
       train_labels = np.array(train_labels)       
       train_data = np.vstack(train_data)
       train_labels = np.hstack(train_labels)
       
       for i in range(len(classifiers)):
           classifiers[i].train(train_data, train_labels)
           acc_train = classifiers[i].accuracy(train_data, train_labels)
           acc_test = classifiers[i].accuracy(test_data, test_label)
           d[names[i]]['train_acc_list'].append(acc_train)
           d[names[i]]['test_acc_list'].append(acc_test)
           classifiers[i].reset()
    
    for name in d.keys():
        d[name]['train_acc'] = np.mean(d[name]['train_acc_list'])
        d[name]['train_std'] = np.std(d[name]['train_acc_list'])
        d[name]['test_acc'] = np.mean(d[name]['test_acc_list'])
        d[name]['test_std'] = np.std(d[name]['test_acc_list'])
        del d[name]['train_acc_list']
        del d[name]['test_acc_list']
    
    return d
               
data_desc, data_label = genere_dataset_gaussian(np.array([1,1]),np.array([[1,0],[0,1]]), \
                                                   np.array([-1,-1]),np.array([[1,0],[0,1]]),\
                                                   10)

print("Le dataset généré avec ", np.shape(data_desc)[0], "lignes de ", np.shape(data_desc)[1], "colonnes.")

# Affichage :
#plot2DSet(data_desc,data_label)

claissfiers = [ClassifierKNN(2,3), ClassifierKNN(2,3)]
d = cross_validation_l(claissfiers, data_desc, data_label, 10)
        
    
    