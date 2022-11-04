import math
import numpy as np
import random
import matplotlib.pyplot as plt

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
            
        return fitness,result

    def crossover(self,w,N,fitness,crossing_rate):
        # Seleciona os pares que realizarão crossover entre si, gerando o indivíduo da geração seguinte. Este indivíduo é armazenado no novo vetor w
        x,y = w[0],w[1]

        crossover_pairs = [random.choices(range(N), weights=fitness, k=2) for iteration in range(math.floor(N/2))]
        alpha = 0
        new_x = []
        new_y = []
        for pair in crossover_pairs:
            alpha = random.random()
            if (random.random()<=crossing_rate):
                new_x.append(alpha*x[pair[0]]+(1-alpha)*x[pair[1]])
                new_y.append(alpha*y[pair[0]]+(1-alpha)*y[pair[1]])

                new_x.append(alpha*x[pair[1]]+(1-alpha)*x[pair[0]])
                new_y.append(alpha*y[pair[1]]+(1-alpha)*y[pair[0]])
            else:
                new_x.append(x[pair[0]])
                new_y.append(y[pair[0]])

                new_x.append(x[pair[1]])
                new_y.append(y[pair[1]])


        #print(f"ox: {['%.1f' % value for value in w[0]]}")
        #print(f"nx: {['%.1f' % value for value in new_w[0]]}\n")
        
        #print(f"oy: {['%.1f' % value for value in w[1]]}")
        #print(f"ny: {['%.1f' % value for value in new_w[1]]}\n")
        
        return np.array([new_x,new_y])

    def mutate(self,w,N,mutation_rate):
        new_x = []
        new_y = []
        for individual in range(N):
            individual_x = w[0][individual]
            individual_y = w[1][individual]
            if (random.random()<=mutation_rate):
                new_x.append(individual_x+random.uniform(-1,1))
                new_y.append(individual_y+random.uniform(-1,1))
                #print(f'mutou o individuo {individual}!')
                #print(f'x de: {w[0][individual]} para {new_x[-1]}!')
                #print(f'y de: {w[1][individual]} para {new_y[-1]}!')
            else:
                new_x.append(individual_x)
                new_y.append(individual_y)
        return np.array([new_x,new_y])

    def generate_samples(self, N):
        return np.random.uniform(low=-10, high=10, size=(2,N)) 

    def run(self,sample_size,n_iterations,crossing_rate,mutation_rate):
        N = sample_size
        w = self.generate_samples(N)
        worse_fit = []
        best_fit = []
        avg_fit = []
        
        for iteration in range(n_iterations):
            # Realizando o ranking linear, retornando um vetor que possui a aptidão de cada indivíduo em relação ao vetor de entrada
            fitness,result = self.linear_ranking(w,N)
            
            worst_index = fitness.index(min(fitness))
            worse_fit.append(result[worst_index])

            best_index = fitness.index(max(fitness))
            best_fit.append(result[best_index])

            avg_fit.append(sum(result)/N)

            w = self.crossover(w,N,fitness,crossing_rate)
            w = self.mutate(w,N,mutation_rate)
        x,y = w[0],w[1]
        #print(f"resultado:  {self.bird_func(w)}")
        print(f"acuracia:  {sum(self.bird_func(w))/(-106.77*N)}")

        plt.plot(worse_fit, 'r')
        plt.plot(best_fit,'g')
        plt.plot(avg_fit,'b')
        plt.show()

        #TODO Criar algoritmo de mutação e especificar algoritmo de crossover   
        return None

if __name__ == "__main__":

    # Inicializando o algoritmo genético
    ga = GeneticAlgorithym()
    ga.run(100,100,0.7,0.01)
