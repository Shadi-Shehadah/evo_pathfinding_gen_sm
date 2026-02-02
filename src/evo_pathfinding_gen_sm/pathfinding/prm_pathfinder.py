import numpy as np
import shapely 
import networkx as nx
import heapq
import yaml

from evo_pathfinding_gen_sm.genetic_algorithm.genetic_code import Genotype, Phenotype
from evo_pathfinding_gen_sm.geometry import Environment

with open("../config/config.yaml",'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
design_rules = cfg['design_rules']
material_properties = cfg['material_properties']
pathfinding_cfg = cfg['prm']
rng = np.random.default_rng()


class PRMPathfinder:
    def __init__(self, environment: Environment, genotype: Genotype = None):
        self.environment = environment
        self.path = []
        self.graph = nx.Graph()
        self.start_point = environment.start_point
        self.target_point = environment.target_point

        self.min_flange_length = design_rules['min_flange_length']
        self.max_flange_length = design_rules['max_flange_length']
        self.min_bend_angle = design_rules['min_bend_angle']


        self.sheet_thickness = material_properties['thickness']

        self.genotype_sensitivity = pathfinding_cfg['genotype_sensitivity']
        self.bend_weight = pathfinding_cfg['bend_weight']
        self.genetic_weight = pathfinding_cfg['genetic_weight']
        self.node_count = pathfinding_cfg['node_count']



        self.open_set = [] 
        self.closed_set = set()



    def add_nodes(self, count=1, sampling_mode='random', max_failed_attempts=100, coordinates=None, genotype: Genotype = None):
        added_nodes = []
        attempts = 0
        max_attempts = max_failed_attempts + count
        while len(added_nodes) < count and attempts < max_attempts:

            
            if sampling_mode == 'random':
                minx, miny, maxx, maxy = self.environment.free_area.bounds
                point = shapely.geometry.Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            
            elif sampling_mode == 'genotype':
                minx, miny, maxx, maxy = self.environment.free_area.bounds
                point = shapely.geometry.Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))

                field_val = genotype.get_field_value(point.x, point.y)

                prob_accept = 1.0 / (1.0 + np.exp(self.genotype_sensitivity * field_val))

                if rng.uniform(0,1) > prob_accept:
                    attempts += 1
                    continue
            



            if self.environment.free_area.contains(point):
                self.graph.add_node(point, pos=(point.x, point.y), target=False, start=False, 
                                    g=float('inf'), h=0, f=float('inf'), 
                                    parent=None, bends=0) 
                added_nodes.append(point)
            attempts += 1
        return added_nodes





    def connect_nodes(self, node1, node2):
        if not self.environment.blocked_area.crosses(shapely.geometry.LineString([node1, node2])):
            if node2 != self.target_point and node1 != self.target_point:
                self.graph.add_edge(node1, node2, weight=node1.distance(node2))
            else:
                target = self.target_point
                end = shapely.Point(self.environment.mount2.plate.coords[1])
                parent = node1 if node1 != target else node2

                # 3. Minimum Bend Angle Check
                vec1 = np.array([target.x - parent.x, target.y - parent.y])
                vec2 = np.array([end.x - target.x, end.y - target.y])

                angle1_rad = np.arctan2(vec1[1], vec1[0])
                angle2_rad = np.arctan2(vec2[1], vec2[0])
                
                # Calculate deviation (bend angle)
                diff_deg = abs(np.degrees(angle1_rad - angle2_rad))
                
                # Normalize to 0-180
                if diff_deg > 180:
                    diff_deg = 360 - diff_deg

                if diff_deg > self.min_bend_angle:
                    self.graph.add_edge(node1, node2, weight=node1.distance(node2))
                    




    def build_roadmap(self, sampling_mode='random', connection_mode='all', n=500, node_count=1000, radius=80,genotype = None):
        self.reset_pathfinder()
        nodes = self.add_nodes(count=node_count, sampling_mode=sampling_mode, genotype=genotype)
        nodes.append(self.start_point)
        nodes.append(self.target_point)
        
        # Initialize start/target with correct attributes
        self.graph.add_node(self.start_point, pos=(self.start_point.x, self.start_point.y), 
                            target=False, start=True, g=0, h=0, f=0, 
                            parent=self.start_point, bends=0) # Parent of start is start (or None)
        
        self.graph.add_node(self.target_point, pos=(self.target_point.x, self.target_point.y), 
                            target=True, start=False, g=float('inf'), h=0, f=float('inf'), 
                            parent=None, bends=0)



        if connection_mode == 'nearestn':
            for node in nodes:
                distances = [(other_node, node.distance(other_node)) for other_node in nodes if other_node != node]
                distances.sort(key=lambda x: x[1])
                for i in range(min(n, len(distances))):
                    if distances[i][1] <= radius:
                        self.connect_nodes(node, distances[i][0])

        if connection_mode == 'radius':
            for node in nodes:
                for other_node in nodes:
                    if other_node != node:
                        dist = node.distance(other_node)
                        if dist <= radius:
                            self.connect_nodes(node, other_node)

        if connection_mode == 'all':
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    self.connect_nodes(nodes[i], nodes[j])
        






    def find_path(self, start_point, target_point,parent_genotype = None):
        # 1. Reset costs for a fresh search
        for node in self.graph.nodes:
            self.graph.nodes[node]["g"] = float('inf')
            self.graph.nodes[node]["f"] = float('inf')
            self.graph.nodes[node]["parent"] = None

        self.graph.nodes[start_point]["g"] = 0
        self.update_h_cost()
        
        self.open_set = []
        # Push Tuple: (f_score, node)
        heapq.heappush(self.open_set, (0, start_point))
        self.closed_set = set()

        while self.open_set:
            # Pop smallest f-value
            current_f, current_node = heapq.heappop(self.open_set)

            if current_node == target_point:
                break # Path found!

            if current_node in self.closed_set:
                continue
            self.closed_set.add(current_node)

            for neighbor in self.graph.neighbors(current_node):
                if neighbor in self.closed_set:
                    continue

                parent_of_current = self.graph.nodes[current_node]["parent"]
                
                if self.validate_node(parent_of_current, current_node, neighbor):
                    
                    # Calculate new costs
                    new_g, new_bends = self.calculate_g_and_bends(parent_of_current, current_node, neighbor,genotype=parent_genotype)
                    
                    if new_g < self.graph.nodes[neighbor]["g"]:
                        self.graph.nodes[neighbor]["g"] = new_g
                        self.graph.nodes[neighbor]["h"] = neighbor.distance(target_point)
                        self.graph.nodes[neighbor]["f"] = new_g + self.graph.nodes[neighbor]["h"]
                        self.graph.nodes[neighbor]["parent"] = current_node
                        self.graph.nodes[neighbor]["bends"] = new_bends
                        
                        heapq.heappush(self.open_set, (self.graph.nodes[neighbor]["f"], neighbor))



        # Reconstruct path
        self.path = []
        
        if self.graph.nodes[target_point]["parent"] is None:
            print("No path found")
            return []

        current = target_point
        while current != start_point:
            self.path.append(current)
            current = self.graph.nodes[current]["parent"]
            # Safety break
            if current is None: 
                print("Error in path reconstruction")
                return []
                
        self.path.append(start_point)
        self.path.reverse()
        return self.path






    def update_h_cost(self):
        for node in self.graph.nodes():
            self.graph.nodes[node]["h"] = node.distance(self.target_point)




    def calculate_g_and_bends(self, parent, current, neighbor,genotype = None):
        
        # 1. Distance Cost
        dist = current.distance(neighbor)
        
        # 2. Parental Alignment cost:
        genetic_cost = 0

        if genotype is not None and hasattr(genotype, 'vector_field'):
                # A. Calculate Move Vector (Normalized)
                move_vec = np.array([neighbor.x - current.x, neighbor.y - current.y])
                move_len = np.linalg.norm(move_vec)
                
                if move_len > 0:
                    move_dir = move_vec / move_len
                    
                    # B. Get Field Vector at Current Location
                    # Assuming vector_field is accessible by grid coordinates [y][x] or similar
                    # We cast to int because grid indices must be integers
                    try:
                        ix, iy = int(current.x), int(current.y)
                        # Shape check to prevent index errors
                        if 0 <= iy < genotype.vector_field.shape[0] and 0 <= ix < genotype.vector_field.shape[1]:
                            field_vec = genotype.vector_field[iy, ix] # Get vector (dx, dy)
                            
                            # Normalize field vector (if not already)
                            field_len = np.linalg.norm(field_vec)
                            
                            if field_len > 0:
                                field_dir = field_vec / field_len
                                
                                # C. Calculate Alignment (Dot Product)
                                # 1.0 = Perfectly Aligned, -1.0 = Opposite, 0.0 = Perpendicular
                                alignment = np.dot(move_dir, field_dir)
                                
                                # D. Convert to Cost
                                # We want alignment (1.0) to be cheap (0.0 cost)
                                # We want misalignment (-1.0) to be expensive (2.0 cost)
                                # Formula: Cost = (1.0 - Alignment) / 2.0  -> Range [0.0, 1.0]
                                genetic_cost = (1.0 - alignment) * dist # Scale by distance traveled
                    except Exception as e:
                        pass


        # 3. Bend Cost
        # Retrieve cumulative bends from current node
        current_bends = self.graph.nodes[current]["bends"]
        is_bend = False
        
        if parent is not None and parent != current:
            # Check if this move creates a new bend
            vec1 = np.array([current.x - parent.x, current.y - parent.y])
            vec2 = np.array([neighbor.x - current.x, neighbor.y - current.y])
            
            # Angle check (simplified: if dot product is not 1, it's a turn)
            angle_rad = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
            angle_deg = abs(np.degrees(angle_rad)) % 180
            
            if angle_deg > 0: 
                is_bend = True
        
        new_bends = current_bends + (1 if is_bend else 0)
        
        # Total G
        new_g = self.graph.nodes[current]["g"] + (dist) + ( (1 if is_bend else 0) * self.bend_weight)  + genetic_cost * self.genetic_weight
        
        return new_g, new_bends




    def validate_node(self, parent, current, neighbor):
        # 1. Start Node Case: Always valid if within range
        if parent is None or parent == current:
            parent = shapely.Point(self.environment.mount1.plate.coords[1])

        # 2. Flange Length Check
        dist = current.distance(neighbor)
        if dist < self.min_flange_length or dist > self.max_flange_length:
            return False

        # 3. Minimum Bend Angle Check
        # Vector 1: Parent -> Current
        vec1 = np.array([current.x - parent.x, current.y - parent.y])
        # Vector 2: Current -> Neighbor
        vec2 = np.array([neighbor.x - current.x, neighbor.y - current.y])

        angle1_rad = np.arctan2(vec1[1], vec1[0])
        angle2_rad = np.arctan2(vec2[1], vec2[0])
        
        # Calculate deviation (bend angle)
        diff_deg = abs(np.degrees(angle1_rad - angle2_rad))
        
        # Normalize to 0-180
        if diff_deg > 180:
            diff_deg = 360 - diff_deg
            
        if diff_deg < self.min_bend_angle:
            return False
            
    
        # Check for self intersection:
        if parent is not None:
            if not self.check_self_collision(current, neighbor):
                return False
                
        return True
    



    def check_self_collision(self, current_node, neighbor):
            """
            Returns False if the new segment (current->neighbor) collides 
            with any part of the path history.
            """
            # 1. Define the segment we are trying to add
            new_segment = shapely.geometry.LineString([current_node, neighbor])
            
            # Define a safety margin (e.g., thickness of your sheet metal)
            # If segments get closer than this, it counts as a collision.
            part_thickness = self.sheet_thickness
            
            # 2. Traverse backwards through the lineage
            cursor = current_node
            
            # We skip the first iteration (immediate parent) because we are 
            # connected to it, so intersection is guaranteed at the vertex.
            skip_check = True 
            
            while True:
                parent = self.graph.nodes[cursor]["parent"]
                
                # Stop if we reached the start
                if parent is None or parent == cursor:
                    break
                
                if skip_check:
                    skip_check = False
                else:
                    if new_segment.crosses(self.environment.mount1.plate) or new_segment.crosses(self.environment.mount2.plate):
                        return False
                    
                    # 3. Create the historical segment
                    old_segment = shapely.geometry.LineString([parent, cursor])
                    
                    # CHECK A: Do they cross strictly? (forming a loop)
                    if new_segment.crosses(old_segment):
                        # print(f"Self-Collision: Crossing detected at {cursor}")
                        return False
                    
                    # CHECK B: Are they too close? (Parallel overlapping or tight folding)
                    # We check distance > thickness. 
                    # Note: distance() finds the shortest distance between two shapes.
                    if new_segment.distance(old_segment) < part_thickness:
                        # print(f"Self-Collision: Parts too close ({new_segment.distance(old_segment):.2f})")
                        return False
                
                # Move backwards
                cursor = parent
                
            return True
    
    def reset_pathfinder(self):
        self.graph.clear()
        self.path = []

    def get_phenotype_from_genotype(self, genotype: Genotype):
        node_count=self.node_count
        self.build_roadmap(node_count=node_count, genotype=genotype)
        self.find_path(self.start_point, self.target_point, parent_genotype=genotype)
        if not self.path or len(self.path) < 2:
            return None
        
        return Phenotype(self.environment, self.path)