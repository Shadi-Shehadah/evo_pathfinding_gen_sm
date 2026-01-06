import numpy as np
import shapely 
import networkx as nx
import heapq
from evo_pathfinding_gen_sm.genetic_algorithm.genetic_code import Genotype, Phenotype
from evo_pathfinding_gen_sm.geometry import Environment, Grid


class GridPathfinder:
    def __init__(self, environment: Environment, genotype: Genotype = None):
        self.environment = environment
        self.grid = environment.grid
        self.graph = environment.grid.graph
        self.start_point = environment.start_point
        self.target_point = environment.target_point
        self.path = []
    
        self.open_set = [self.start_point]
        self.closed_set = []


    def find_path(self,start_point, target_point):
        #self.path = nx.astar_path(self.graph, source= start_point, target= target_point, weight='weight')

        self.graph.nodes[start_point]["g"] = 0
        self.update_h_cost()
        heapq.heapify(self.open_set)

        
        while self.open_set:
            current_node = heapq.heappop(self.open_set)

 
            # Adding the immediate neighbors:
            # Because they are too close, adding them as a new bend would never be valid
            # so they are only added if they have a valid connection to the parent of the parent
            for neighbor in self.graph.neighbors(current_node):
                if self.validate_immediate_neighbor(current_node,neighbor):
                    g = self.find_g_cost(neighbor, current_node)
                    if g < self.graph.nodes[neighbor]["g"]:
                        self.graph.nodes[neighbor]["g"] = g
                        self.graph.nodes[neighbor]["f"] = g + self.graph.nodes[neighbor]["h"]
                        self.graph.nodes[neighbor]["parent"] = current_node
                        if neighbor not in self.open_set:
                            heapq.heappush(self.open_set, neighbor)
            self.closed_set.append(current_node)






        return self.path
    








    def update_h_cost(self):
        for node in self.graph.nodes():
            self.graph.nodes[node]["h"] = node.distance(self.target_point)









    def validate_immediate_neighbor(self, current_node,neighbor):
        
        prev = self.graph.nodes[current_node]["parent"]


        # check for line of sight between the neighbor and the parent of the current node:
        if not self.environment.line_of_sight(neighbor,prev):
            return False
        
        
        # Check for minimum bend angle:
        vec1 = np.array([neighbor.x,neighbor.y]) - np.array(prev.x,prev.y)
        vec2 = np.array([prev.x,prev.y]) - np.array(self.graph.nodes[prev]["parent"].x,self.graph.nodes[prev]["parent"].y)

        angle1_rad = np.arctan2(vec1[1], vec1[0])
        angle2_rad = np.arctan2(vec2[1], vec2[0])

        if (abs(np.degrees(angle1_rad-angle2_rad)) % 180) < 35:
            return False
        

        # check for self intersection: WIP
        #current_node = node
        #total_part = shapely.LineString()
        #while current_node != self.graph.nodes[current_node]["parent"]:
        #    if shapely.geometry.LineString([node, self.graph.nodes[node]["parent"]]).intersection(shapely.geometry.LineString([current_node, self.graph.nodes[current_node]["parent"]])):
        #        return False
        #    current_node = self.graph.nodes[current_node]["parent"]


        # Check for minimum/maximum flange length: 
        if neighbor.distance(prev) < self.minimum_flange_length or neighbor.distance(prev) > self.maximum_flange_length:
            return False
        
        return True
    

        # Check for bend collision: TO-DO (maybe)











    def find_g_cost(self,neighbor,current_node):
        pass

        # finding number of bends:
        bend_count = 0
        current = current_node
        while self.graph.nodes[current]["parent"] is not current:
            bend_count += 1
            current = self.graph.nodes[current]["parent"]
        bend_weight = 10
        
        #finding total lenght:
        length = neighbor.distance(current_node)
        length_weight = 1
        g = self.graph.nodes[current_node]["g"] + (length * length_weight) + (bend_count * bend_weight)
        return g