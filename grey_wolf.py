import numpy as np
from random import random,choice


def euclidean_metrics(x,y):
    distance = np.sqrt(np.sum((x-y)**2,axis=1))
    return distance

def fitness_function(X_train,centroids):
    j = 0
    for x in X_train:
        dist = euclidean_metrics(x,centroids)
        j += np.sum(dist)
    return j 

class GWO:
    def __init__(self,n_clusters,population,alpha_p=2,alpha_k=0,choice_type='kmeans++',const_population=False,centroids=[],max_iter=1000) -> None:
        self.n_clusters = n_clusters
        self.population = population
        self.const_population = const_population
        if self.const_population == True:
            self.centroids = centroids
        self.f_changes = []
        self.choice_type = choice_type
        self.j = np.inf
        self.alpha_p = alpha_p
        self.alpha_k = alpha_k

    def fit_centroids(self,X_train):
        if self.const_population == False:
            if self.choice_type == 'standard':
                min_,max_ = np.min(X_train,axis=0), np.max(X_train,axis=0)
                self.centroids =np.array([[np.random.uniform(min_,max_) for i in range(self.n_clusters)] for j in range(self.population)])
            if self.choice_type == 'kmeans++':
                self.centroids = []
                for _ in range(self.population):
                    centers = [np.zeros(X_train[0].shape) for c in range(self.n_clusters)]
                    centers[0] = [choice(X_train)]
                    centers_temp = []
                    for i in range(self.n_clusters-1):
                        centers_temp.append(centers[i])
                        dists = np.sum([euclidean_metrics(cent,X_train) for cent in centers_temp],axis=0)
                        dists /= np.sum(dists)
                        new_centroid_idx = np.random.choice(range(len(X_train)), size=1,p=dists)
                        centers[i+1] = X_train[new_centroid_idx]
                    self.centroids.append(np.reshape(np.array(centers),(self.n_clusters,X_train.shape[1])))
                self.centroids = np.array(self.centroids)
        iter = 0
        self.first_centers = self.centroids
        f_values = []
        for center in self.centroids:
            f_values.append(fitness_function(X_train,center))
        f_val = np.min(f_values)
        if f_val < self.j:
            self.j = f_val
        self.f_changes.append(self.j)
        self.best_centers = self.centroids[np.argmin(f_values)]
        while iter < self.max_iter:
            sorted_pop = sorted(zip(f_values,self.centroids))
            alpha = self.alpha_p - ((self.alpha_p-self.alpha_k)*iter)/self.max_iter
            wolfs_adb = np.array([[np.zeros(X_train[0].shape) for i in range(self.n_clusters)] for _ in range(3)])
            # for i in range(3):
            #     r_1 = random()
            #     r_2 = random()
            #     a = 2 * alpha * r_1 - alpha
            #     c = 2 * r_2
            #     sorted_pop[i][1] = (sorted_pop[i][1] - a*(c*sorted_pop[i][1]-sorted_pop[i][1]))
            # for i in range(3,self.n_clusters):
            #     sorted_pop[i][1] = (sorted_pop[0][1] + sorted_pop[1][1] + sorted_pop[2][1])/3
            for i in range(3):
                r_1 = random()
                r_2 = random()
                a = 2 * alpha * r_1 - alpha
                c = 2 * r_2
                wolfs_adb[i] = sorted_pop[i][1]
                sorted_pop[i][1] = (sorted_pop[i][1] - a*(c*sorted_pop[i][1]-sorted_pop[i][1]))
            
            for i in range(3,self.n_clusters):
                r_1 = random()
                r_2 = random()
                a = 2 * alpha * r_1 - alpha
                c = 2 * r_2
                x_1 = wolfs_adb[0] - a*(c*wolfs_adb[0]-sorted_pop[i][1])
                x_2 = wolfs_adb[1] - a*(c*wolfs_adb[0]-sorted_pop[i][1])
                x_3 = wolfs_adb[2] - a*(c*wolfs_adb[0]-sorted_pop[i][1])
                sorted_pop[i][1] = (x_1 + x_2 + x_3)/3

            self.centroids = np.array([sorted_pop[j][1] for j in range(self.population)])
            f_values = []
            for center in self.centroids:
                 f_values.append(fitness_function(X_train,center))
            f_val = np.min(f_values)
            if f_val < self.j:
                self.j = f_val
                self.best_centers = self.centroids[np.argmin(f_values)]
            self.f_changes.append(self.j)
            iter += 1
            
    def predict(self,X_train):
        centroids_idx = []
        for x in X_train:
            dist = euclidean_metrics(x,self.best_centers)
            idx = np.argmin(dist)
            centroids_idx.append(idx)
        return  centroids_idx