from inspect import indentsize
import math
import numpy as np
import pandas as pd
import random

class Genetics():
    
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def bird_func(self, x, y):
        #aplicando bird function
        return np.sin(x)*np.exp(np.power(1-np.cos(y), 2))+np.cos(y)*np.exp(np.power(1-np.sin(x), 2))+np.power((x-y), 2)

    def linear_ranking(self, x):
        #computing derivative to the Sigmoid function
        return()

    def generate_matrix(self, N):
        w = np.random.uniform(low=-10, high=10, size=(2,N))
        x,y = w[0],w[1]
        
        iterations = 10000
        
        while(iterations != 0):
            bird_y = np.array(self.bird_func(x,y))
            bird_map = {}
            
            for pos in range(N):
                bird_map[pos] = bird_y[pos]
            ranking_map = sorted(bird_map.items(), key=lambda x:x[1])
            
            start = 1
            end = N
            
            # Geram as aptidões dos indivíduos e as organiza de acordo com a numeração de cada indivíduo
            aptidoes = [(ranking_map[rank][0],math.floor(start+(end-start)*(N-1-rank)/(N-1))) for rank in range(N)]
            aptidoes.sort()
        
            # Lista os indíviduos do problema e os seus respectivos pesos/aptidões
            individuals = [i[0] for i in aptidoes]
            weights = [i[1] for i in aptidoes]
            
            # Seleciona os pares que realizarão crossover entre si, gerando o indivíduo da geração seguinte. Este indivíduo é armazenado no novo vetor w
            chosen_pairs = [random.choices(individuals, weights=weights, k=2) for individual in individuals]
            w = np.array([[(w[variable][pair[0]]+w[variable][pair[1]])/2 for pair in chosen_pairs] for variable in [0,1]])
            x,y = w[0],w[1]
            
            iterations=iterations-1
        
        print(f"x:  {x.sum()/x.size}")
        print(f"y:  {y.sum()/y.size}")

        #TODO Criar algoritmo de mutação e especificar algoritmo de crossover   

        






if __name__ == "__main__":

    #initializing the neuron class
    genetics = Genetics()

    genetics.generate_matrix(5)
