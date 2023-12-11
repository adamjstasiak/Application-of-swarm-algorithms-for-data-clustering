import numpy as np
from random import random,choice


def euclidean_metrics(x,y):
    """
        function calculating euclidean metrics 
        between point and other point or set of points
    """
    return np.sqrt(np.sum((x-y)**2,axis=1))

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

class GWO:
    """
        Class representing Grey wolf optimizer
    """
    def __init__(self,n_clusters: int,population: int,alpha_p=2,alpha_k=0,choice_type='kmeans++',const_population=False,centroids=[],max_iter=1000) -> None:
        """
        Initial object attributes:
            n_clusters - number of cluster in data_set
            population - size of wolf pack
            const_population - set algorithm to work from specific first population mostly for iteration test
            centroids - current population
            choice_type - detetrmine how we choose first population 
            alpha_p - highest value range of parameter alpha
            alpha_k - lowest value  range of parameter alpha
            max_iter - number of iteration 
            j - current best value of fitness function
            f_changes - set which rember values of fitness funtion in each iteration 
        """
        self.n_clusters = n_clusters
        self.population = population
        self.const_population = const_population
        if self.const_population == True:
            self.centroids = centroids
        self.choice_type = choice_type
        self.alpha_p = alpha_p
        self.alpha_k = alpha_k
        self.max_iter = max_iter
        self.j = np.inf
        self.f_changes = []
        self.best_clusters = []

    def fit_centroids(self,X_train):
        """
            Method which calculate best centers for given dataset
        """
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
                    """ Reshape for keeping proper size in population """
                    self.centroids.append(np.reshape(np.array(centers),(self.n_clusters,X_train.shape[1])))

        self.centroids = np.array(self.centroids)
        iter = 0
        """
            Calculation of fitness function values for each solution in population
            then we remember the best value and solution 
        """
        f_values = []
        for center in self.centroids:
            f_values.append(fitness_function(X_train,center))
        f_val = np.min(f_values)
        if f_val < self.j:
            self.j = f_val
        self.f_changes.append(self.j)
        self.best_clusters = self.centroids[np.argmin(f_values)]
        while iter < self.max_iter:
            """ 
                Main loop of the algorithm firstly it short solution based on their fitness function value.
                initial table for store position of wolfs alfa, beta and delta the best three solutions.
                Also it calculate current value of parameter alpha 
            """
            sorted_pop = list(dict(sorted({f_values[i]: self.centroids[i] for i in range(self.population)}.items())).values())
            wolfs_adb = np.array([sorted_pop[i] for i in range(3)])
            alpha = self.alpha_p - ((self.alpha_p-self.alpha_k)*iter)/self.max_iter
            # for i in range(3):
            #     r_1 = random()
            #     r_2 = random()
            #     a = 2 * alpha * r_1 - alpha
            #     c = 2 * r_2
            #     sorted_pop[i][1] = (sorted_pop[i][1] - a*(c*sorted_pop[i][1]-sorted_pop[i][1]))
            # for i in range(3,self.n_clusters):
            #     sorted_pop[i][1] = (sorted_pop[0][1] + sorted_pop[1][1] + sorted_pop[2][1])/3
            """
                In this section algorithm calculates a and c parameters for alfa, beta and delta, and update their posotions 
            """
            for i in range(3):
                r_1 = random()
                r_2 = random()
                a = 2 * alpha * r_1 - alpha
                c = 2 * r_2
                sorted_pop[i]= sorted_pop[i] - a * (c*sorted_pop[i]-sorted_pop[i])
            """
                In this section we calculate a and c for rest of wolfes and update their position based on 
                current position of alfa, beta and delta.
            """
            for i in range(3,self.population):
                r_1 = random()
                r_2 = random()
                a = 2 * alpha * r_1 - alpha
                c = 2 * r_2
                x_1 = wolfs_adb[0] - a*(c*wolfs_adb[0]-sorted_pop[i])
                x_2 = wolfs_adb[1] - a*(c*wolfs_adb[1]-sorted_pop[i])
                x_3 = wolfs_adb[2] - a*(c*wolfs_adb[2]-sorted_pop[i])
                sorted_pop[i]= (x_1 + x_2 + x_3)/3
            """
                Calculation of fitnes function value for new solutions and update of current best solution a function value 
            """
            self.centroids = sorted_pop
            f_values = []
            for center in self.centroids:
                 f_values.append(fitness_function(X_train,center))
            f_val = np.min(f_values)
            if f_val < self.j:
                self.j = f_val
                self.best_clusters = self.centroids[np.argmin(f_values)]
            self.f_changes.append(self.j)
            iter += 1
            
    def predict(self,X_train):
        """ 
            Simple classification, it assign centroid for points based on their distance to the center.
        """
        centroids_idx = []
        for x in X_train:
            dist = euclidean_metrics(x,self.best_clusters)
            idx = np.argmin(dist)
            centroids_idx.append(idx)
        return  centroids_idx