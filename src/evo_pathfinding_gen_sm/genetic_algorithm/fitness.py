import numpy as np
import shapely.geometry
import yaml

from evo_pathfinding_gen_sm.genetic_algorithm.genetic_code import Genotype, Phenotype
from evo_pathfinding_gen_sm.geometry import Environment
from evo_pathfinding_gen_sm.structural_simulator import StructuralSimulator

with open("../config/config.yaml",'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
fitness_cfg = cfg['fitness']




def find_fitness(candidate: Phenotype):
    simulator = StructuralSimulator(candidate)
    score = 100

    # Subtract for bends:
    score -= (len(candidate.bends)   *  fitness_cfg['bend_cost'])
    #print('the number of bends is:' + str(len(candidate.bends)))


    # Subtract for mass:
    score -= (candidate.mass  * fitness_cfg['mass_cost'])
    #print('the mass is:' + str(candidate.mass))


    # Subtract for deflection:
    score -= (simulator.calculate_max_deflection()    *  fitness_cfg['deflection_cost'])
    #print('the max deflection under 100N of vertical force is:' + str(candidate.mass))


    return score /100
