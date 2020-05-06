# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:21:52 2020

@author: hp
"""
import numpy as np

def cross_validation(classifiers, data, labels, k=10):
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
    # train_acc = []
    # test_acc = []
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
# ------------------------ A COMPLETER :
def leave_one_out(C, DS):
    """ Classifieur * tuple[array, array] -> float
    """
    X, Y = DS
    count = 0
    n = len(X)
    for i in range(len(X)):
        X_ = np.delete(X, i, axis=0)
        Y_ = np.delete(Y, i, axis=0)
        
        C.train(X_, Y_)
        if C.predict(X[i]) == Y[i] :
            count += 1
    return count / n
        
        
            