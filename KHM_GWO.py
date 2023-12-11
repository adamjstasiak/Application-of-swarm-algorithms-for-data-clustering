import numpy as np
from random import random,choice


def euclidean_metrics(x,y):
    distance = np.sqrt(np.sum((x-y)**2,axis=1))
    return distance

def fitness_fuction(X_train,centroids,alpha):
    j = 0
    for x in X_train:
        distance = euclidean_metrics(x,centroids)
        j += len(centroids)/np.sum(1/(distance**alpha))
    return j

class KHMGWO:
    """
        Class representing Grey wolf optimizer
    """
    def __init__(self,n_clusters: int,population: int,alpha_p=2,alpha_k=0,alpha=2,const_population=False,centroids=[],max_iter=1000) -> None:
        """
        Initial object attributes:
            n_clusters - number of cluster in data_set
            population - size of wolf pack
            const_population - set algorithm to work from specific first population mostly for iteration test
            centroids - current population

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
        self.alpha_p = alpha_p
        self.alpha_k = alpha_k
        self.max_iter = max_iter
        self.j = np.inf
        self.f_changes = []
        self.alpha = alpha
    def fit_centroids(self,X_train):
        """
            Method which calculate best centers for given dataset
        """
        if self.const_population == False:
            min_,max_ = np.min(X_train,axis=0), np.max(X_train,axis=0)
            self.centroids =np.array([[np.random.uniform(min_,max_) for i in range(self.n_clusters)] for j in range(self.population)])
           
        iter = 0
        """
            Calculation of fitness function values for each solution in population
            then we remember the best value and solution 
        """
        f_values = []
        for center in self.centroids:
            f_values.append(fitness_fuction(X_train,center,self.alpha))
        f_val = np.min(f_values)
        if f_val < self.j:
            self.j = f_val
        self.f_changes.append(self.j)
        self.best_centers = self.centroids[np.argmin(f_values)]
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

            self.centroids = sorted_pop
            for j in range(self.population):
                memberships = [[] for _ in range(self.n_clusters)]
                wheights = []
                for x in X_train:
                    distance = euclidean_metrics(x,self.centroids[i])
                    membership_ = self.membership(distance)
                    for i in range(self.n_clusters):
                        memberships[i].append(membership_[i])
                        wheights.append(self.wage(distance))
                new_centroids = self.recalculate_centroids(memberships,wheights,X_train)
                self.centroids[i] = new_centroids
            f_values = []
            for center in self.centroids:
                 f_values.append(fitness_fuction(X_train,center,self.alpha))
            f_val = np.min(f_values)
            if f_val < self.j:
                self.j = f_val
                self.best_centers = self.centroids[np.argmin(f_values)]
            self.f_changes.append(self.j)
            iter += 1

    def membership(self,distance):
        member = []
        for i in range(len(distance)):
            l = np.power(distance[i],-self.alpha-2)
            m = np.sum(np.power(distance,-self.alpha-2))
            member.append(l/m)
        return member

    def wage(self,distance):
        l = np.sum(np.power(distance,-self.alpha-2))
        m = np.power(np.sum(np.power(distance,-self.alpha)),2)
        return l/m

    def recalculate_centroids(self,memberships,wheigts,X_train):
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

    def predict(self,X_train):
        clasification = []
        for x in X_train:
            distance = euclidean_metrics(x,self.best_centers)
            membership_ = self.membership(distance)
            idx = np.argmax(membership_)
            clasification.append(idx)
        return clasification
    