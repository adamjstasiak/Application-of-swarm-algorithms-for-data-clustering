# khajit has wares if you have coin

import numpy as np
from random import choices,random
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
    def __init__(self,n_clusters,smp,cdc,srd,c,population_size,max_velocity=10,proportion=50,spc=False,max_iter=1000) -> None:
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
        self.population_size = population_size
        self.max_velocity = max_velocity
        self.f_changes = []
        

    def fit_cntroids(self,X_train):
        if self.cdc > len(X_train[0]):
            raise(ValueError)
        min_,max_ = np.min(X_train,axis=0), np.max(X_train,axis=0)
        
        # self.centroids = np.array([np.array([np.random.uniform(min_,max_) for i in range(self.n_clusters)]) for _ in range(self.population_size)])
        self.centroids = np.array([np.random.uniform(min_,max_) for i in range(self.n_clusters)])

        iter = 0
        f_values = []
        for el in self.centroids:
             f_values.append(fitness_function(X_train,el))
        if np.min(f_values) < self.j:
            self.j = np.min(f_values)
        self.f_changes.append(self.j)
        self.best_centers = self.centroids[np.argmin(f_values)]
        # self.velocity = np.array([[np.zeros(X_train[0].shape) for i in range(self.n_clusters)] for _ in range(self.population_size)])
        self.velocity = [np.zeros(X_train[0].shape) for i in range(self.n_clusters)]
        self.rest_cat = np.array(choices(self.centroids,k=np.int((self.n_clusters*self.proportion)/100)))
        self.rest_idx = []
        self.hunt_idx = []
        for i in range(len(self.centroids)):
            if self.centroids[i] not in self.rest_cat:
                self.hunt_idx.append(i)
            else:
                self.rest_idx.append(i) 
        while iter < self.max_iter:
            # prev_centroids = self.centroids
            self.hunting()
            self.rest_cat = self.resting(X_train)
            for i in range(len(self.rest_idx)):
                self.centroids[self.rest_idx[i]]  = self.rest_cat[i]

            f_values = []
            for el in self.centroids:
                f = fitness_function(X_train,el)
                f_values.append(f)
            if np.min(f_values) < self.j:
                self.j = np.min(f_values)
                self.best_centers = self.centroids[np.argmin(f_values)]
            self.f_changes.append(self.j)
            self.rest_cat = np.array(choices(self.centroids,k=np.int((self.n_clusters*self.proportion)/100)))
            self.rest_idx = []
            self.hunt_idx = []
            for i in range(len(self.centroids)):
                if self.centroids[i] not in self.rest_cat:
                    self.hunt_idx.append(i)
                else:
                    self.rest_idx.append(i) 
            iter += 1



    def resting(self,X_train):
        # new_cats = np.array([[np.zeros(X_train[0].shape) for i in range(self.n_clusters)] for _ in range(np.int((self.n_clusters*self.proportion)/100))])
        new_cats = [np.zeros(X_train[0].shape) for i in range(len(self.rest_cat))] 
        for i in range(len(self.rest_cat)):
            candidates = [self.rest_cat[i] for _ in range(self.smp)]
            for j in range(self.smp):
                candidate = candidates[j]
                for el in candidate:
                    for k in range(self.cdc):
                        oper = np.random.choice(Operations)
                        if oper == Operations.plus:
                            el[k] += (el[k]*self.srd)
                        if oper == Operations.minus:
                            el[k] -= (el[k]*self.srd) 
                candidates[j] = candidate
            if self.spc == True:
                candidates.append(self.rest_cat[i])
            f_values = [fitness_function(X_train,can) for can in candidates]
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
        for idx in self.hunt_idx:
            r = random()
            self.velocity[idx] = self.velocity[idx] + r * self.c * (self.best_centers- self.centroids[idx])
            self.velocity[idx] = np.where(self.velocity[idx]< self.max_velocity,self.velocity[idx],self.max_velocity)
            self.centroids[idx] = self.centroids[idx] + self.velocity[idx]
         
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

