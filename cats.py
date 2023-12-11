# khajit has wares if you have coin

import numpy as np
from random import choices,random,choice,sample
from enum import Enum 

def euclidean_metrics(x,centroids):
    """
        function calculating euclidean metrics 
        between point and other point or set of points
    """
    return  np.sqrt(np.sum((x - centroids)**2,axis=1))

def fitness_function(X_train,centroids):
    """
        Function to calculate our objective function
        It sums distance metrics between each point
        In dataset and given centroids.

        Parameters: 

            X_train - dataset
            centroids - current centroids coordinates

        output:

            j - objective function value
    """
    
    j = 0
    for x in X_train:
        dist = euclidean_metrics(x,centroids)
        j += np.sum(dist)
    return j 

class CSO:
    """
        Class representing Cat Swarm Optimization algorithm
    """
    def __init__(self,n_clusters: int,smp: int,cdc: int,srd: float,c:float,spc=False,choice_type='kmeans++',const_population=False,centroids=[],max_velocity=10,proportion=25,max_iter=1000) -> None:
        """
            Initial object attributes:
                n_clusters - number of cluster in data_set
                smp - seeking memory pool - number of memory in which seeking cat should improve it
                cdc - counts of dimension change - number of features in dataset algorithm change
                srd - seeking range of the selected dimension - percent value of each resting we change current position
                spc - self position consideration - boolean parameter which decide if we keep current cat postion as new solution
                c - main value we consider in calculate new velocity for hunting cat
                choice_type - detetrmine how we choose first population
                const_population - set algorithm to work from specific first population mostly for iteration test
                centroids - current population
                max_velocity - max value of which hunting cat can update his current position
                max_iter - number of iteration 
                j - current best value of fitness function
                f_changes - set which rember values of fitness funtion in each iteration 
                proportion - number of cats in rest mode
        """
        
        self.n_clusters = n_clusters
        self.smp = smp
        self.cdc = cdc
        self.srd = srd
        self.spc = spc
        self.c = c
        self.choice_type = choice_type
        if self.spc == True:
            self.smp = smp-1
        self.const_population = const_population
        if self.const_population == True:
            self.centroids = centroids
        self.max_velocity = max_velocity
        self.max_iter = max_iter       
        self.j = np.inf
        self.f_changes = []
        self.proportion = np.int((self.n_clusters*proportion)/100)
        self.best_clusters = []

    def fit_centroids(self,X_train):
        """
            Method which calculate best centers for given dataset
        """
        if self.cdc > len(X_train[0]):
            raise(ValueError)
        if self.const_population == False:
            """
                if parameter const_population is false we choose  first population based on to methods:
                standard - its typical for kmeans algorithm to choose first centroid, we choose first points 
                randomly based on uniform normal distribution.
                kmeans++ - first we select first center randomly next points we choose based on distance between point 
                and previously choosen centroids. 
            """
            if self.choice_type == 'standard':
                min_,max_ = np.min(X_train,axis=0), np.max(X_train,axis=0)
                self.centroids = np.array([np.random.uniform(min_,max_) for i in range(self.n_clusters)])
            if self.choice_type == 'kmeans++':
                self.centroids = [np.zeros(X_train.shape[0]) for _ in range(self.n_clusters)] 
                self.centroids[0] = [choice(X_train)]
                centroid_temp = []
                for i in range(self.n_clusters-1):
                    centroid_temp.append(self.centroids[i])
                    dists = np.sum([euclidean_metrics(centroid,X_train) for centroid in centroid_temp],axis=0)
                    dists /= np.sum(dists)
                    new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=dists)
                    self.centroids[i+1] = X_train[new_centroid_idx]
                self.centroids = np.array(self.centroids)
                self.centroids = np.reshape(self.centroids,(self.n_clusters,X_train.shape[1]))
        iter = 0
        """
            Calculation of fitness function values for each solution in population
            then we remember the best value and solution 
        """
        f_values = fitness_function(X_train,self.centroids)
        if f_values < self.j:
            self.j = f_values
        self.f_changes.append(self.j)
        self.best_clusters = self.centroids
        self.velocity = np.zeros(self.centroids.shape)
        self.rest_cat, self.rest_idx,self.hunt_idx = self.choice_rest_cat()

        while iter < self.max_iter:
            """
                Main loop of the algorithm its calculate new position of cats based on their mode
            """
            self.hunting()
            self.rest_cat = self.resting(X_train)
            for i in range(len(self.rest_idx)):
                self.centroids[self.rest_idx[i]]  = self.rest_cat[i]
            """
                Calculation of fitnes function value for new solutions and update of current best solution a function value 
            """    
            f_values = fitness_function(X_train,self.centroids)
            self.f_changes.append(self.j) 
            if f_values < self.j:
                self.j = f_values
                self.best_clusters = self.centroids
            
            self.rest_cat, self.rest_idx,self.hunt_idx = self.choice_rest_cat()

            iter += 1

    def choice_rest_cat(self):
        """Function which randomly select cats to resting mode

            :param self: attributes of class
            :return: rest_cat - array of select solution rest_idx - indexes of selected cats  
        """
        hunt_idx = [i for i in range(self.n_clusters)]
        rest_idx = sample(hunt_idx,self.proportion)
        rest_cat = []
        for idx in rest_idx:
            rest_cat.append(self.centroids[idx])
            hunt_idx.remove(idx)
        return np.array(rest_cat,dtype='float32'),rest_idx,hunt_idx
    
    def resting(self,X_train):
        """
            Resting mode of CSO, it creates smp copies of each cat and the randomly add or substract
            srd value of cat in cdc dimensions. Next choose new cat randomly with probablity based of their distance to each point in dataset
        """
        new_cats = np.zeros(self.rest_cat.shape,dtype='float')
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
            f_values = [sum(euclidean_metrics(cand,X_train)) for cand in candidates]
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
        """
            Hunting modes, calculates cat velocity then updates cat position
        """
        r = random()
        self.velocity = self.velocity + r * self.c * (self.best_clusters-self.centroids)
        self.velocity = np.where(self.velocity< self.max_velocity,self.velocity,self.max_velocity)
        self.centroids = self.centroids + self.velocity 
            
    def predict(self,X_train):
        """ 
            Simple classification, it assign centroid for points based on their distance to the center.
        """
        centroids_idx = []
        for x in X_train:
            dist = euclidean_metrics(x,self.best_clusters)
            idx = np.argmin(dist)
            centroids_idx.append(idx)
        return centroids_idx


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

