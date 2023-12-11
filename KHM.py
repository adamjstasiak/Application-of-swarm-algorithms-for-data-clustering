import numpy as np

def euclidean_metrics(x,y):
    distance = np.sqrt(np.sum((x-y)**2,axis=1))
    return distance

def fitness_fuction(X_train,centroids,alpha):
    j = 0
    for x in X_train:
        distance = euclidean_metrics(x,centroids)
        j += len(centroids)/np.sum(1/(distance**alpha))
    return j


class KHM:
    def __init__(self,n_clusters = 4,max_iter=1000,alpha=2.0,centroids=[]) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.alpha = alpha
        self.j = np.inf
        self.f_changes = []
        self.centroids = np.array(centroids)

    def fit_centroids(self,X_train):
        min_,max_ = np.min(X_train,axis = 0),np.max(X_train,axis=0)
        self.centroids = np.array([np.random.uniform(min_,max_) for i in range(self.n_clusters)])
        
        iteration = 0
        self.j = fitness_fuction(X_train,self.centroids,self.alpha)
        self.f_changes.append(self.j)

        while self.max_iter > iteration:
            memberships = [[] for _ in range(self.n_clusters)]
            wheights = [] 
            for x in X_train:
                distance = euclidean_metrics(x,self.centroids)
                for el in distance:
                    if el == 0:
                        el = np.finfo(float).eps
                membership_ = self.membership(distance)
                for i in range(self.n_clusters):                  
                    memberships[i].append(membership_[i])
                wheight = self.wage(distance)
                wheights.append(wheight)
            new_centroids = self.recalculate_centroids(memberships,wheights,X_train)
            new_j = fitness_fuction(X_train,new_centroids,self.alpha)
            if new_j < self.j:
                self.j = new_j
                self.centroids = new_centroids
            self.f_changes.append(self.j)
            iteration += 1

    def membership(self,distance) -> list:
        member = []
        for i in range(len(distance)):
            l = np.power(distance[i],-self.alpha-2)
            m = np.sum(np.power(distance,-self.alpha-2))
            member.append(l/m)
        return member

    def wage(self,distance) -> float:
        l = np.sum(np.power(distance,-self.alpha-2))
        m = np.power(np.sum(np.power(distance,-self.alpha)),2)
        return l/m

    def recalculate_centroids(self,memberships,wheigts,X_train) -> np.array:
        centroids = []
        for k in range(self.n_clusters):
            l = np.zeros(X_train[0].shape)
            m = 0
            for i in range(X_train.shape[0]):  
                l += (memberships[k][i]*wheigts[i] * X_train[i])
                m += (memberships[k][i]*wheigts[i])
            center = l/m
            centroids.append(center)
        return np.array(centroids)

    def predict(self,X_train) -> list:
        clasification = []
        for x in X_train:
            distance = euclidean_metrics(x,self.centroids)
            membership_ = self.membership(distance)
            idx = np.argmax(membership_)
            clasification.append(idx)
        return clasification
    