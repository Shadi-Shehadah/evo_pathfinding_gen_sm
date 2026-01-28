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
rng = np.random.default_rng()


class GridPathfinder:
    def __init__(self, environment: Environment, genotype: Genotype = None):
        self.environment = environment
        self.path = []
        self.grid = environment.grid
        self.graph = self.grid.graph
        self.start_point = environment.start_point
        self.target_point = environment.target_point

        self.min_flange_length = design_rules['min_flange_length']
        self.max_flange_length = design_rules['max_flange_length']
        self.min_bend_angle = design_rules['min_bend_angle']




        self.sheet_thickness = material_properties['thickness']


        self.genetic_weight = 1
        self.bend_weight = 1

        # Priority Queue: Stores (f_cost, tie_breaker, node_object)
        self.open_set = []
        self.closed_set = set()
        self.counter = 0 # Unique ID for tie-breaking







    def find_path(self, start_point, target_point,parent_genotype = None):
        # 1. Reset costs for a fresh search
        for node in self.graph.nodes:
            self.graph.nodes[node]["g"] = float('inf')
            self.graph.nodes[node]["f"] = float('inf')
            self.graph.nodes[node]["parent"] = None
            self.graph.nodes[node]["bends"] = 0

        self.graph.nodes[start_point]["g"] = 0
        self.graph.nodes[start_point]["bends"] = 0 
        self.graph.nodes[start_point]["parent"] = start_point

        self.update_h_cost(parent_genotype)
        
        self.open_set = []

        
        # Push Tuple: (f_score, node)
# FIX 2: Use a counter for tie-breaking
        # Tuple structure: (f_score, unique_counter, node_object)
        self.counter = 0 
        heapq.heappush(self.open_set, (0, self.counter, start_point))
        self.counter += 1

        self.closed_set = set()

        while self.open_set:
            current_f, _, current_node = heapq.heappop(self.open_set)

            if current_node == target_point:
            # Final Validation: Is the last segment long enough?
                parent = self.graph.nodes[target_point]["parent"]
                final_segment_len = parent.distance(target_point)
    
                if final_segment_len < self.min_flange_length:
            # Path invalid: Ended with a tiny stump. 
            # Don't break; continue searching for a better path.
                    continue 
        
                break # Path found and Valid!


            if current_node in self.closed_set:
                continue
            self.closed_set.add(current_node)

            # Get references for the "Standard" path
            parent_node = self.graph.nodes[current_node]["parent"]
            
            # Get references for the "Shortcut" path (Theta*)
            # The "Grandparent" is needed to check the angle at the Parent if we skip Current
            grandparent_node = None
            if parent_node is not None:
                grandparent_node = self.graph.nodes[parent_node]["parent"]

            for neighbor in self.graph.neighbors(current_node):
                if neighbor in self.closed_set:
                    continue
                
                # --- PATH 1: THETA* SHORTCUT (Try connecting Parent -> Neighbor) ---
                # Conditions: 
                # 1. We have a parent (can't shortcut from start).
                # 2. Line of Sight exists between Parent and Neighbor.
                # 3. The bend at Parent is valid (using Grandparent -> Parent -> Neighbor).
                shortcut_success = False
                
                if parent_node is not None and parent_node != current_node:
                     if self.environment.line_of_sight(parent_node, neighbor):
                        if self.validate_node(grandparent_node, parent_node, neighbor):
                            
                            # Calculate cost from PARENT, skipping current
                            # Note: pass 'parent_node' as current to your cost function effectively
                            new_g, new_bends = self.calculate_g_and_bends(
                                grandparent_node, parent_node, neighbor, genotype=parent_genotype
                            )
                            
                            if new_g < self.graph.nodes[neighbor]["g"]:
                                self.graph.nodes[neighbor]["g"] = new_g
                                self.graph.nodes[neighbor]["h"] = neighbor.distance(target_point)
                                self.graph.nodes[neighbor]["f"] = new_g + self.graph.nodes[neighbor]["h"]
                                self.graph.nodes[neighbor]["parent"] = parent_node # Point to Parent!
                                self.graph.nodes[neighbor]["bends"] = new_bends
                                
                                heapq.heappush(self.open_set, (self.graph.nodes[neighbor]["f"], self.counter, neighbor))
                                self.counter += 1
                                shortcut_success = True

                # --- PATH 2: STANDARD A* (Connect Current -> Neighbor) ---
                # Only runs if shortcut failed or wasn't possible
                if not shortcut_success:
                    if self.validate_node(parent_node, current_node, neighbor):
                        
                        new_g, new_bends = self.calculate_g_and_bends(
                            parent_node, current_node, neighbor, genotype=parent_genotype
                        )
                        
                        if new_g < self.graph.nodes[neighbor]["g"]:
                            self.graph.nodes[neighbor]["g"] = new_g
                            self.graph.nodes[neighbor]["h"] = neighbor.distance(target_point)
                            self.graph.nodes[neighbor]["f"] = new_g + self.graph.nodes[neighbor]["h"]
                            self.graph.nodes[neighbor]["parent"] = current_node # Point to Current
                            self.graph.nodes[neighbor]["bends"] = new_bends
                            
                            heapq.heappush(self.open_set, (self.graph.nodes[neighbor]["f"], self.counter, neighbor))
                            self.counter += 1
            



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








    def update_h_cost(self, genotype=None):
            # Tuning Parameter:
            # Increase this to make the pathfinder follow the genotype field more aggressively.
            # Decrease to prioritize the shortest straight line.
            genetic_weight = 10.0 

            for node in self.graph.nodes():
                # 1. Physical Distance (Euclidean)
                dist_geom = node.distance(self.target_point)
                
                # 2. Genetic Cost
                dist_genetic = 0.0
                if genotype is not None:
                    # Retrieve the field value at this node's coordinates
                    # Assuming genotype.get_field_value(x, y) exists
                    raw_val = genotype.get_field_value(node.x, node.y)
                    dist_genetic = raw_val * genetic_weight

                # 3. Combine
                # Logic: 
                # If raw_val is Negative (Valley), h decreases -> Node looks "closer" -> Explored sooner.
                # If raw_val is Positive (Hill), h increases -> Node looks "farther" -> Avoided.
                total_h = dist_geom + dist_genetic
                
                # Safety: A* heuristics generally shouldn't be negative.
                self.graph.nodes[node]["h"] = max(0.0, total_h)





    def calculate_g_and_bends(self, parent, current, neighbor, genotype=None):
            
            # 1. Distance Cost
            dist = current.distance(neighbor)
            
            # 2. Parental Alignment Cost (Vector Field)
            genetic_cost = 0.0
            
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
                        # Fallback if indices are out of bounds or field is missing
                        pass

            # 3. Bend Cost
            # Retrieve cumulative bends from current node
            current_bends = self.graph.nodes[current]["bends"]
            is_bend = False
            
            if parent is not None and parent != current:
                vec1 = np.array([current.x - parent.x, current.y - parent.y])
                vec2 = np.array([neighbor.x - current.x, neighbor.y - current.y])
                
                angle_rad = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
                angle_deg = abs(np.degrees(angle_rad)) % 180
                
                # Using your tolerance or > 0
                if angle_deg > self.min_bend_angle: 
                    is_bend = True
            
            new_bends = current_bends + (1 if is_bend else 0)
            
            # 4. Total G
            step_bend_cost = (self.bend_weight if is_bend else 0)
            
            new_g = self.graph.nodes[current]["g"] + dist + step_bend_cost + (genetic_cost * self.genetic_weight)
            
            return new_g, new_bends







    def validate_node(self, parent, current, neighbor):
        # 1. Start Node Case
        if parent is None or parent == current:
            return True

        # 2. Calculate Angle
        bend_angle = self.calculate_bend_angle(parent, current, neighbor)
        
        # 3. Classify Move (Drifting vs Turning)
        is_going_straight = bend_angle < self.min_bend_angle 
        
        # --- CASE A: GOING STRAIGHT ---
        if is_going_straight:
            # Check Max Length (only strict rule for straight lines)
            # Use backtracking to get the true length if you want to be strict,
            # but usually immediate parent check is enough for A* to prune.
            if parent.distance(neighbor) > self.max_flange_length:
                return False
            return True

        # --- CASE B: TURNING ---
        else:
            # 1. Self-Collision (Always check)
            if not self.check_self_collision(current, neighbor):
                return False            

            # 2. Standard Flange Length Check (For the rest of the path)
            incoming_length = parent.distance(current)
            
            # Quick Check
            if incoming_length >= self.min_flange_length:
                return True 
                
            # Deep Check (Backtracking)
            true_length = self.get_accumulated_length(current, parent)
            
            if true_length < self.min_flange_length:
                return False # Too short, and not near an endpoint

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




    



    def calculate_bend_angle(self, p_prev, p_curr, p_next):
        """ Calculates the deviation angle in degrees (0=Straight, 90=Turn) """
        v_in = np.array([p_curr.x - p_prev.x, p_curr.y - p_prev.y])
        v_out = np.array([p_next.x - p_curr.x, p_next.y - p_curr.y])
        
        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)
        
        if norm_in < 1e-9 or norm_out < 1e-9: return 0.0 # Treat as straight
        
        dot = np.dot(v_in, v_out)
        cos_theta = np.clip(dot / (norm_in * norm_out), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))





    def get_accumulated_length(self, current_node, parent_node):

        total_length = current_node.distance(parent_node)
        
        # Cursors for backtracking
        child = current_node
        curr = parent_node
        
        # Look back up to 50 steps (approx 25mm, enough to satisfy min_flange)
        for _ in range(50):
            # 1. Get Predecessor
            # Note: We must access the graph to find the history
            if curr not in self.graph.nodes: break
            pred = self.graph.nodes[curr]["parent"]
            
            # Stop if we hit Start or End of chain
            if pred is None or pred == curr: 
                break
                
            # 2. Check Alignment
            # Is the segment (Pred -> Curr -> Child) straight?
            angle = self.calculate_bend_angle(pred, curr, child)
            
            # Using your 'Drifting Straight' tolerance
            if angle < self.min_bend_angle: 
                # It is effectively one line, add length and keep going back
                total_length += curr.distance(pred)
                child = curr
                curr = pred
            else:
                # We hit a bend! The straight segment started here.
                break
                
        return total_length
    


    def get_phenotype_from_genotype(self, genotype: Genotype):
        self.find_path(self.start_point, self.target_point, parent_genotype=genotype)
        return Phenotype(self.environment, self.path)