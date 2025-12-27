import numpy as np
import shapely 
import sys
from pathlib import Path
from src.genetic_algorithm.genetic_code import Genotype, Phenotype
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry import Environment
from genetic_algorithm.genetic_code import Phenotype, Genotype

class GridPathfinder:
    def __init__(self,environment: Environment):
        self.environment = environment
        self.grid = environment.grid
        self.path = []
    
    def find_path(self,start_point, end_point):
        # Placeholder for pathfinding logic
        self.path = [start_point, end_point]
        return self.path