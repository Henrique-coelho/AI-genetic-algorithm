from inspect import indentsize
import math
import numpy as np
import pandas as pd
import random

class GeneticAlgorithym():
    
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def bird_func(self, w):
        #aplicando bird function
        x,y = w[0],w[1]
        return np.sin(x)*np.exp(np.power(1-np.cos(y), 2))+np.cos(y)*np.exp(np.power(1-np.sin(x), 2))+np.power((x-y), 2)

    def linear_ranking(self, w, N):
        # Realizando o ranking linear, retornando um vetor que possui a aptidão de cada indivíduo em relação ao vetor de entrada
        result = np.array(self.bird_func(w))
        result_map = {}
            
        # Organiza as posições dos pares x e y de forma crescente de suas respostas
        for pos in range(N):
            result_map[pos] = result[pos]
        ranking_map = sorted(result_map.items(), key=lambda x:x[1])
            
        # Posições minimas e máximas
        pos_min_result = 1
        pos_max_result = N
            
        # Geram as aptidões dos indivíduos e as organiza de acordo com a numeração de cada indivíduo
        fitness_aux = [(ranking_map[rank][0],math.floor(pos_min_result+(pos_max_result-pos_min_result)*(N-1-rank)/(N-1))) for rank in range(N)]
        fitness_aux.sort()
        fitness = [i[1] for i in fitness_aux]
            
        return fitness

    def crossover(self,w,N,fitness):
        # Seleciona os pares que realizarão crossover entre si, gerando o indivíduo da geração seguinte. Este indivíduo é armazenado no novo vetor w
        crossover_pairs = [random.choices(range(N), weights=fitness, k=2) for iteration in range(N)]
        new_w = np.array([[(w[variable][pair[0]]+w[variable][pair[1]])/2 for pair in crossover_pairs] for variable in [0,1]])
        
        #print(f"ox: {['%.1f' % value for value in w[0]]}")
        #print(f"nx: {['%.1f' % value for value in new_w[0]]}\n")
        
        #print(f"oy: {['%.1f' % value for value in w[1]]}")
        #print(f"ny: {['%.1f' % value for value in new_w[1]]}\n")
        
        return new_w
           
    def generate_samples(self, N):
        return np.random.uniform(low=-10, high=10, size=(2,N)) 

    def run(self,sample_size,n_iterations):
        N = sample_size
        w = self.generate_samples(N)
        x,y = w[0],w[1]
        
        for iteration in range(n_iterations):
            # Realizando o ranking linear, retornando um vetor que possui a aptidão de cada indivíduo em relação ao vetor de entrada
            fitness = self.linear_ranking(w,N)
            w = self.crossover(w,N,fitness)
        
        print(f"x:  {x.sum()/x.size}")
        print(f"y:  {y.sum()/y.size}")

        #TODO Criar algoritmo de mutação e especificar algoritmo de crossover   
        return None

if __name__ == "__main__":

    # Inicializando o algoritmo genético
    ga = GeneticAlgorithym()
    ga.run(5,1)
