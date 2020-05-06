# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 00:02:53 2020

@author: hp
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from node import NoeudCategoriel

def classe_majoritaire(Y):
    values, counts = np.unique(Y, return_counts=True)
    return values[counts.argmax()]
def shannon(P):
    e = 0
    base = len(P)
    if base == 1:
        return 0.0
    for p in P:
        if p == 0: continue
        e -= p * math.log(p, base)
    return e

def plot_shannon():
    P = np.linspace(0, 1, 5000)
    values = []
    for p in P:
        values.append(shannon([p, 1-p]))
    plt.plot(P, values)
    plt.xlabel('p')
    plt.ylabel('shannon([p, 1-p])')
    plt.show()

def entropie(labels):
    _, counts = np.unique(labels, return_counts=True)
    P = counts / sum(counts)
    return shannon(P)
    

elections = pd.read_csv('../tme-07/data/elections.csv')
election_data = np.array(elections.drop('Label', axis=1).values)
election_label = np.array(elections['Label'])
att = elections.columns[:-1]

def construit_AD(X, Y, epsilon, LNoms=[]):
    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        ############################# DEBUT ########
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.
        
        ## COMPLETER ICI !
        
        ############################# FIN ######## 
        Xbest_valeurs = []
        for i in range(len(LNoms)):
          values, counts = np.unique(X[:, i], return_counts=True)
          pAtt = counts / sum(counts)
          # entropy conditionnelle
          e = 0
          for val in range(len(values)):
             mask = X[:, i] == values[val]
             indicies = [k for k in range(len(mask)) if mask[k] == True]
             e+= entropie(Y[indicies]) * pAtt[val]
          if min_entropie >= e:
            Xbest_valeurs = values
            min_entropie = e
            i_best = i
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud
    
    
def condithional_entropy(X, Y, attributs=[0,1,2]):
    global att
    X_best = None
    i_best = -1
    ent = 1.1
    for at in attributs:
        print(att[at])
        values, counts = np.unique(X[:, at], return_counts=True)
        print(values)
        print(counts)
        pAtt = counts/ sum(counts)
        e = 0
        for value in range(len(values)):
            mask = (X[:, at] == values[value])
            # extracting indices
            print(Y[mask])
            e += entropie(Y[mask])* pAtt[value]
            print(e)
        print(e)
        if e <= ent:
            X_best = values
            i_best = at
            ent = e

        print('IG({0})= {1}'.format(att[at], entropie(Y)-e))
    print(X_best, ent, i_best)


            
    
    
    