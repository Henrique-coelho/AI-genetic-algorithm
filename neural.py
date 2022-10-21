import numpy as np
import pandas as pd


class Genetics():
    
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def bird_func(self, x, y):
        #applying the sigmoid function,
        return np.sin(x)*np.exp(np.power(1-np.cos(y), 2))+np.cos(y)*np.exp(np.power(1-np.sin(x), 2))+np.power((x-y), 2)

    def linear_ranking(self, x):
        #computing derivative to the Sigmoid function
        return()

    def generate_matrix(self, N):
        w = np.random.uniform(low=-10, high=10, size=(2,N))
        x,y = w[0],w[1]

        bird_y = np.array(self.bird_func(x,y))
        bird_map = {}

        for i in range(len(bird_y)):
            bird_map[i] = bird_y[i]
        ranking_map = sorted(bird_map.items(), key=lambda x:x[1])[::-1]

        #print(ranking_map)

        mini = min(bird_y)
        maxi = max(bird_y)
        np.sort(bird_y)
        aptidoes = []
        for i in range(N):
            aptidoes.append((ranking_map[i][0],(mini+(maxi-mini)*(N-i-1)/(N-1))))
        print(aptidoes)

        sum_apt = sum([i[1] for i in aptidoes])
        prob_individuo = [(i[0],i[1]/sum_apt) for i in aptidoes]
        print(prob_individuo)


        #TODO Avaliar a ordem de importancia das aptidoes       

        






if __name__ == "__main__":

    #initializing the neuron class
    genetics = Genetics()

    genetics.generate_matrix(5)
