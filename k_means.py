import numpy as np

def euclidean_metrics(x,y):
    distnace =np.sqrt(np.sum((x-y)**2,axis=1)) 
    return distnace
    
def fitness_function(X_train,centroids):
    j = 0
    for x in X_train:
        dist = euclidean_metrics(x,centroids)
        j += np.sum(dist)
    return j 
class KMeans:
    def __init__(self,n_clusters=4,max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.j = np.inf
        self.f_changes = []
        self.best_clusters = []

    def fit_centroids(self,X_train):
        min_,max_ = np.min(X_train,axis = 0),np.max(X_train,axis=0)
        self.centroids = [np.random.uniform(min_,max_) for i in range(self.n_clusters)]
        iteration = 0 
        f_val = fitness_function(X_train,self.centroids)
        if f_val < self.j:
            self.j = f_val
        self.f_changes.append(self.j)
        self.best_clusters = self.centroids
        while iteration < self.max_iter:

            sort_points = [[] for i in range(self.n_clusters)]
            for x in X_train:
                distance = euclidean_metrics(x,self.centroids)
                centroid_idx = np.argmin(distance)
                sort_points[centroid_idx].append(x)
            self.centroids = [np.mean(cluster,axis=0) for cluster in sort_points]    
            

            for i , centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = self.best_clusters[i]
            f_val = fitness_function(X_train,self.centroids)
            if f_val < self.j:
                self.j = f_val
                self.best_clusters = self.centroids
            self.f_changes.append(self.j)
            iteration += 1
        
    def predict(self,X_train):
        centroids_idx = []
        for x in X_train:
            dist = euclidean_metrics(x,self.best_clusters)
            idx = np.argmin(dist)
            centroids_idx.append(idx)
        return  centroids_idx    


