import numpy as np
import shapely.geometry
import networkx as nx
import yaml
import copy

from evo_pathfinding_gen_sm.genetic_algorithm.genetic_code import Genotype, Phenotype
from evo_pathfinding_gen_sm.geometry import Environment
from evo_pathfinding_gen_sm.genetic_algorithm.genetic_processes import mutate_shift, mutate_invert, uniform_crossover
from evo_pathfinding_gen_sm.pathfinding.prm_pathfinder import PRMPathfinder
from evo_pathfinding_gen_sm.pathfinding.grid_pathfinder import GridPathfinder
from evo_pathfinding_gen_sm.genetic_algorithm.fitness import find_fitness


with open("../config/config.yaml",'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
genetic_cfg = cfg['genetic_algorithm']

rng = np.random.default_rng()




class Evolution_Manager():
    def __init__(self,environment : Environment):
        self.environment = environment
        self.generations = []
        self.current_generation_index = -1




    def initialize(self,pathfinder_type: str, population_size: int):
        start_point = self.environment.start_point
        target_point = self.environment.target_point
        self.generations = []
        self.pathfinder = PRMPathfinder(self.environment)
        

        initial_generation = Generation([])
        while len(initial_generation.phenotypes) < population_size:
            self.pathfinder.build_roadmap(node_count=genetic_cfg['prm_initial_nodes'])
            path = self.pathfinder.find_path(start_point, target_point)
            if path is not None and len(path) >= 2:
                phenotype = Phenotype(self.environment,path)
                initial_generation.phenotypes.append(phenotype)

        initial_generation.update_fitnesses()
        self.generations.append(initial_generation)
        self.current_generation_index = 0
        if pathfinder_type == 'Grid':
            self.pathfinder = GridPathfinder(self.environment)




    def new_generation(self,previous_generation: Generation, selection_type= 'roulette', population_size: int = None):
        selection_list = copy.deepcopy(previous_generation.phenotypes)
        ages = [phenotype.age for phenotype in selection_list]
        new_generation_phenotypes = []


        if selection_type == 'roulette' or selection_type == 'rank':
            if selection_type == 'roulette':
                fitness_sum = sum(previous_generation.fitnesses)
                probabilities = [fitness / fitness_sum for fitness in previous_generation.fitnesses]
                cumulative_probabilities = np.cumsum(probabilities)
                
            elif selection_type == 'rank':
                sorted_indices = np.argsort(previous_generation.fitnesses)
                ranks = np.empty_like(sorted_indices)
                ranks[sorted_indices] = np.arange(len(previous_generation.fitnesses))
                rank_sum = sum(ranks)
                probabilities = [rank / rank_sum for rank in ranks]
                cumulative_probabilities = np.cumsum(probabilities)

  

            while len(new_generation_phenotypes) < population_size:
                if rng.random() > genetic_cfg['grid_based_probability']:
                    self.pathfinder = PRMPathfinder(self.environment)
                else:
                    self.pathfinder = GridPathfinder(self.environment)
                    
                if rng.random() < genetic_cfg['crossover_rate']:
                    parent1 = np.searchsorted(cumulative_probabilities, rng.random())
                    parent2 = np.searchsorted(cumulative_probabilities, rng.random())

                    child_genotype = uniform_crossover(selection_list[parent1].genotype, selection_list[parent2].genotype)
                    child_phenotype1 = self.pathfinder.get_phenotype_from_genotype(child_genotype[0])
                    child_phenotype2 = self.pathfinder.get_phenotype_from_genotype(child_genotype[1])
                    if child_phenotype1 is None or child_phenotype2 is None:
                        continue
                    new_generation_phenotypes.append(child_phenotype1)
                    new_generation_phenotypes.append(child_phenotype2)
                    
                    #print(f" Crossover between parents {parent1} and {parent2}")

                elif rng.random() < genetic_cfg['mutation_rate']:
                    parent = selection_list[np.searchsorted(cumulative_probabilities, rng.random())].genotype
                    if rng.binomial(1, genetic_cfg['mutation_shift_probability']) == 1:
                        child_genotype = mutate_shift(parent)
                    else:
                        child_genotype = mutate_invert(parent)

                    child_phenotype = self.pathfinder.get_phenotype_from_genotype(child_genotype)
                    if child_phenotype is None:
                        continue
                    new_generation_phenotypes.append(child_phenotype)
                    #print(f" Mutation applied")
                
                # Else, directly copy a selected phenotype
                else:
                    
                    selected_index = np.searchsorted(cumulative_probabilities, rng.random())
                    selected_phenotype = copy.deepcopy(selection_list[selected_index])
                    selected_phenotype.age += 1
                    #print(f" Direct copy of phenotype {selected_index} with age {selected_phenotype.age}")
                    if selected_phenotype.age > genetic_cfg['max_age']:
                        # If too old, reject
                        continue
                    new_generation_phenotypes.append(selected_phenotype)
                for phenotype in new_generation_phenotypes:
                    phenotype.fitness = find_fitness(phenotype)
                    if phenotype.fitness < genetic_cfg['min_fitness_threshold']:
                        new_generation_phenotypes.remove(phenotype)

        
        new_generation = Generation(new_generation_phenotypes)
        new_generation.update_fitnesses()
        #print("new generation length: " + str(len(new_generation.phenotypes)))
        return new_generation


    def run_evolution(self, generations_count: int, population_size: int, pathfinder_type: str, selection_type: str,fitness_offset: int=0):
        self.initialize(pathfinder_type, population_size)
        print(f"Generation {self.current_generation_index} completed.")
        print(f" Best fitness: {self.generations[self.current_generation_index].best_phenotype.fitness + fitness_offset}")

        #for i, phenotype in enumerate(self.generations[self.current_generation_index].phenotypes):
            #print(f" Phenotype {i} fitness: {phenotype.fitness}")

        for _ in range(generations_count):
            previous_generation = self.generations[self.current_generation_index]
            new_generation = self.new_generation(previous_generation=previous_generation, selection_type=selection_type, population_size=population_size)
            self.generations.append(new_generation)
            self.current_generation_index += 1
            print(f"Generation {self.current_generation_index} completed.")
            print(f" Best fitness: {new_generation.best_phenotype.fitness + fitness_offset}")
            #for i, phenotype in enumerate(self.generations[self.current_generation_index].phenotypes):
                #print(f" Phenotype {i} fitness: {phenotype.fitness}")

        





class Generation():
    def __init__(self,phenotypes =[]):
        self.phenotypes = phenotypes
        for phenotype in self.phenotypes:
            phenotype.fitness = find_fitness(phenotype)   
        self.fitnesses = [phenotype.fitness for phenotype in self.phenotypes]     

        self.best_phenotype = None
        if len(self.phenotypes) > 0:
            best_index = np.argmax(self.fitnesses)
            self.best_phenotype = self.phenotypes[best_index]
        
    def update_fitnesses(self):
        self.fitnesses = []
        for phenotype in self.phenotypes:
            phenotype.fitness = find_fitness(phenotype)   
            self.fitnesses.append(phenotype.fitness)     

        self.best_phenotype = None
        if len(self.phenotypes) > 0:
            best_index = np.argmax(self.fitnesses)
            self.best_phenotype = self.phenotypes[best_index]