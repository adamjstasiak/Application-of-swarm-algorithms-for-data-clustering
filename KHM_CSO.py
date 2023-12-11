import numpy as np
from random import choices,random,choice,sample
from enum import Enum 

def euclidean_metrics(x,centroids):
    """
        function calculating euclidean metrics 
        between point and other point or set of points
    """
    return  np.sqrt(np.sum((x - centroids)**2,axis=1))

def fitness_fuction(X_train,centroids,alpha):
    """
        Function to calculate objective function
        it sums harmonic means of the distance
    """
    j = 0
    for x in X_train:
        distance = euclidean_metrics(x,centroids)
        j += len(centroids)/np.sum(1/(distance**alpha))
    return j
class KHMCSO:
    
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
    def __init__(self,n_clusters,smp,cdc,srd,c,spc=False,alpha=2,const_population=False,centroids=[],max_velocity=10,proportion=25,max_iter=1000) -> None:
        self.n_clusters = n_clusters
        self.smp = smp
        self.cdc = cdc
        self.srd = srd
        self.spc = spc
        self.c = c
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
        self.alpha = alpha
        
    def fit_centroids(self,X_train):
        """
            Method which calculate best centers for given dataset
        """
        if self.cdc > len(X_train[0]):
            raise(ValueError)
        if self.const_population == False:
            min_,max_ = np.min(X_train,axis=0), np.max(X_train,axis=0)
            self.centroids = np.array([np.random.uniform(min_,max_) for i in range(self.n_clusters)])
            self.centroids = np.reshape(self.centroids,(self.n_clusters,X_train.shape[1]))
        iteration = 0
        f_values = fitness_fuction(X_train,self.centroids,self.alpha)
        if f_values < self.j:
            self.j = f_values
        self.f_changes.append(self.j)
        self.best_centers = self.centroids
        self.velocity = np.zeros(self.centroids.shape)
        self.rest_cat, self.rest_idx,self.hunt_idx = self.choice_rest_cat()
        while iteration < self.max_iter:
            self.hunting()
            self.rest_cat = self.resting(X_train)
            for i in range(len(self.rest_idx)):
                self.centroids[self.rest_idx[i]] = self.rest_cat[i]
            f_values + fitness_fuction(X_train,self.centroids,self.alpha)
            if f_values < self.j:
                self.j = f_values
                self.best_centers = self.centroids
            self.f_changes.append(self.j)
            self.rest_cat, self.rest_idx,self.hunt_idx = self.choice_rest_cat()
            iteration += 1
            memberships = [[] for _ in range(self.n_clusters)]
            wheights = []
            for x in X_train:
                distance = euclidean_metrics(x,self.centroids)
                membership_ = self.membership(distance)
                for i in range(self.n_clusters):
                    memberships[i].append(membership_[i])
                wheights.append(self.wage(distance))
            self.centroids = self.recalculate_centroids(memberships,wheights,X_train)
            f_values = fitness_fuction(X_train,self.centroids,self.alpha)
            if f_values < self.j:
                self.j = f_values
                self.best_centers = self.centroids
            self.f_changes.append(self.j)

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
        self.velocity = self.velocity + r * self.c * (self.best_centers-self.centroids)
        self.velocity = np.where(self.velocity< self.max_velocity,self.velocity,self.max_velocity)
        self.centroids = self.centroids + self.velocity 
            

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

