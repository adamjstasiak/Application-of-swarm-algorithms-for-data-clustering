import numpy as np

def euclidean_metrics(x,y):
    distnace =np.sqrt(np.sum((x-y)**2,axis=1)) 
    return distnace
    
class K_Means:
    def __init__(self,n_clusters=4,max_iter=1000) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit_centroids(self,X_train):
        min_,max_ = np.min(X_train,axis = 0),np.max(X_train,axis=0)

        self.centroids = [np.random.uniform(min_,max_) for i in range(self.n_clusters)]
      

        iter = 0 
        prev_centroid = None
        
        while np.not_equal(self.centroids, prev_centroid).any() and iter < self.max_iter:

            sort_points = [[] for i in range(self.n_clusters)]
            for x in X_train:
                distance = euclidean_metrics(x,self.centroids)
                centroid_idx = np.argmin(distance)
                sort_points[centroid_idx].append(x)
                
            prev_centroid =self.centroids
            self.centroids =  [np.mean(cluster,axis=0) for cluster in sort_points]

            for i , centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i]= prev_centroid[i]
            iter += 1
            
    def predict(self,X_train):
        centroids = []
        centroids_idx = []
        for x in X_train:
            dist = euclidean_metrics(x,self.centroids)
            idx = np.argmin(dist)
            centroids.append(self.centroids[idx])
            centroids_idx.append(idx)
        return centroids, centroids_idx    


