import numpy as np
import shapely.geometry
import networkx as nx
import yaml
import copy

from evo_pathfinding_gen_sm.genetic_algorithm.genetic_code import Genotype
from evo_pathfinding_gen_sm.geometry import Environment, Blob

with open("../config/config.yaml",'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
genetic_cfg = cfg['genetic_algorithm']

rng = np.random.default_rng()

def mutate_shift(genotype: Genotype):
    new_genotype = copy.deepcopy(genotype)
    blob_index = rng.integers(len(new_genotype.blobs))
    mutation_distance = genetic_cfg['mutation_distance']

    # 1. Mutate the Blob (DNA)
    new_genotype.blobs[blob_index].x += np.random.uniform(-mutation_distance, mutation_distance)  
    new_genotype.blobs[blob_index].y += np.random.uniform(-mutation_distance, mutation_distance)  
    
    # 2. Update the Vector Field (Phenotype)
    # This ensures the h-cost sees the new blob positions
    new_genotype.update_vector_field()
    
    return new_genotype

def mutate_invert(genotype: Genotype):
    new_genotype = copy.deepcopy(genotype)
    blob_index = rng.integers(len(new_genotype.blobs))
    
    # 1. Mutate Strength
    new_genotype.blobs[blob_index].strength = genotype.blobs[blob_index].strength * -1
    
    # 2. Update Field
    new_genotype.update_vector_field()
    
    return new_genotype





def uniform_crossover(parent1 : Genotype, parent2 : Genotype):
    # Deep copy the lists of blobs so we don't affect parents
    chromosome1 = copy.deepcopy(parent1.blobs)
    chromosome2 = copy.deepcopy(parent2.blobs)
    shorter_chromosome_length = min(len(chromosome1), len(chromosome2))
    p = 0.5

    for i in range(shorter_chromosome_length):
        # Swap genes at index i
        if rng.binomial(1, p):
            temp = copy.deepcopy(chromosome2[i])
            chromosome2[i] = copy.deepcopy(chromosome1[i])
            chromosome1[i] = copy.deepcopy(temp)

        
        if rng.binomial(1, p):
            j = shorter_chromosome_length - i - 1
            if j >= 0: # Safety check
                temp = copy.deepcopy(chromosome2[j])
                chromosome2[j] = copy.deepcopy(chromosome1[j])
                chromosome1[j] = copy.deepcopy(temp)
    
    # Create Child 1
    child1 = Genotype(bends=[], weights=[])
    child1.blobs = chromosome1
    child1.update_vector_field() # <--- Recalculate field based on mixed blobs

    # Create Child 2
    child2 = Genotype(bends=[], weights=[])
    child2.blobs = chromosome2
    child2.update_vector_field() # <--- Recalculate field based on mixed blobs

    return child1, child2