# khajit has wares if you have coin

import numpy as np
from random import choices,random,choice,sample
from enum import Enum 

def euclidean_metrics(x,y):
    distance = np.sqrt(np.sum((x-y)**2,axis=1))
    return distance

def fitness_function(X_train,centroids):
    j = 0
    for x in X_train:
        dist = euclidean_metrics(x,centroids)
        j += np.sum(dist)
    return j 

class CSO:
    def __init__(self,n_clusters,smp,cdc,srd,c,choice_type='kmeans++',max_velocity=10,proportion=25,spc=False,max_iter=1000) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.j = np.inf
        self.spc = spc
        self.smp = smp
        if self.spc == True:
            self.smp = smp-1
        self.cdc = cdc
        self.srd = srd
        self.c = c
        self.proportion = proportion
        self.max_velocity = max_velocity
        self.f_changes = []
        self.choice_type = choice_type
        

    def fit_centroids(self,X_train):
        if self.cdc > len(X_train[0]):
            raise(ValueError)
        if self.choice_type == 'standard':
            min_,max_ = np.min(X_train,axis=0), np.max(X_train,axis=0)
            self.centroids = np.array([np.random.uniform(min_,max_) for i in range(self.n_clusters)])
        if self.choice_type == 'kmeans++':
            self.centroids = [choice(X_train)]
            self.pop_ration = np.int((self.n_clusters*self.proportion)/100)
            for _ in range(self.n_clusters-1):
                dists = np.sum([euclidean_metrics(centroid, X_train) for centroid in self.centroids], axis=0)
                dists /= np.sum(dists)
                new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
                self.centroids += [X_train[new_centroid_idx]]
            self.centroids = np.array(self.centroids)
        iter = 0
        f_values = fitness_function(X_train,self.centroids)
        if f_values < self.j:
            self.j = f_values
        self.f_changes.append(self.j)
        self.best_centers =self.centroids
        self.velocity = [np.zeros(X_train[0].shape) for i in range(self.n_clusters)]
        self.rest_cat, self.rest_idx = self.choice_rest_cat()
        while iter < self.max_iter:
            self.hunting()
            self.rest_cat = self.resting(X_train)
            for i in range(len(self.rest_idx)):
                self.centroids[self.rest_idx[i]]  = self.rest_cat[i]
            f_values = fitness_function(X_train,self.centroids)
            if f_values < self.j:
                self.j = f_values
                self.best_centers = self.centroids
            self.f_changes.append(self.j) 
            self.rest_cat, self.rest_idx = self.choice_rest_cat()
            iter += 1

    def choice_rest_cat(self):
        indexes = sample([i for i in range(self.n_clusters)],self.pop_ration)
        rest_cat = []
        for idx in indexes:
            rest_cat.append(self.centroids[idx])
        return np.array(rest_cat),indexes
    
    def resting(self,X_train):
        new_cats = [np.zeros(X_train[0].shape) for _ in range(len(self.rest_cat))] 
        for i  in range(len(self.rest_cat)):
            candidates = [self.rest_cat[i] for _ in range(self.smp)]
            for cand in candidates:
                for k in range(self.cdc):
                    oper = np.random.choice(Operations,p=(0.51,0.49))
                    if oper == Operations.plus:
                        cand[k] += (cand[k]*self.srd)
                    if oper == Operations.minus:
                        cand[k] -= (cand[k]*self.srd)
            if self.spc == True:
                candidates.append(self.rest_cat[i])
            f_values = np.zeros((1,len(candidates)))
            for x in X_train:
                distance = euclidean_metrics(x,candidates)
                f_values += distance
            if allEqual(f_values) == True:
                probabilty = [1 for _ in range(len(candidates))]
            else:
                probabilty = []
                F_max = np.max(f_values)
                F_min = np.min(f_values)
                for val in f_values:
                    Pj = np.abs(val-F_max)/(F_max-F_min)
                    probabilty.append(Pj)
            mask = choices([i for i in range(len(candidates))],weights=probabilty)
            new_cat = candidates[mask[0]]
            new_cats[i] = new_cat
        return new_cats

    def hunting(self):
        for _ in range(self.n_clusters-self.pop_ration):
            r = random()
            self.velocity = self.velocity + r * self.c * (self.best_centers - self.centroids)
            self.velocity = np.where(self.velocity< self.max_velocity,self.velocity,self.max_velocity)
            self.centroids = self.centroids + self.velocity
    
    def predict(self,X_train):
        centroids = []
        centroids_idx = []
        for x in X_train:
            dist = euclidean_metrics(x,self.best_centers)
            idx = np.argmin(dist)
            centroids.append(self.best_centers[idx])
            centroids_idx.append(idx)
        return centroids, centroids_idx


class Operations(Enum):
    plus  = 1
    minus = 2

def allEqual(list):
    iterator = iter(list)
    try:
        firstItem = next(iterator)
    except StopIteration:
        return True
    for x in iterator:
        if x != firstItem:
            return False
    return True    

