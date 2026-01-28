import numpy as np
import shapely.geometry
import networkx as nx
import heapq

from evo_pathfinding_gen_sm.genetic_algorithm.genetic_code import Genotype
from evo_pathfinding_gen_sm.geometry import Environment

class GridPathfinder:
    def __init__(self, environment: Environment, genotype: Genotype = None):
        self.environment = environment
        self.grid = environment.grid
        self.graph = environment.grid.graph
        self.start_point = environment.start_point
        self.target_point = environment.target_point
        self.path = []
        
        # Constraints
        self.minimum_flange_length = 5.0
        self.maximum_flange_length = 60.0
        self.min_bend_angle = 35.0 # Degrees

        # Priority Queue: Stores (f_cost, tie_breaker, node_object)
        self.open_set = []
        self.closed_set = set()
        self.counter = 0 # Unique ID for tie-breaking

    def find_path(self, start_point, target_point):
        # --- DEBUG START ---
        if start_point not in self.graph:
            print(f"CRITICAL ERROR: Start Node {start_point} is NOT in the graph!")
            return []
        
        degree = self.graph.degree[start_point]
        print(f"DEBUG: Start Node Degree: {degree}")
        if degree == 0:
            print("FAILURE: Start Node is isolated (0 neighbors). Connect it to the grid!")
            return []
        # --- DEBUG END ---

        
        # 1. Reset Graph Costs
        for node in self.graph.nodes:
            self.graph.nodes[node]["g"] = float('inf')
            self.graph.nodes[node]["f"] = float('inf')
            self.graph.nodes[node]["parent"] = None

        # 2. Initialize Start
        self.graph.nodes[start_point]["g"] = 0
        self.graph.nodes[start_point]["h"] = start_point.distance(target_point)
        self.graph.nodes[start_point]["f"] = self.graph.nodes[start_point]["h"]
        self.graph.nodes[start_point]["parent"] = start_point # Start is its own parent (or None)

        self.push_to_open(start_point, self.graph.nodes[start_point]["f"])
        self.closed_set = set()

        # 3. Main Loop
        while self.open_set:
            current_f, _, current_node = heapq.heappop(self.open_set)

            if current_node == target_point:
                return self.reconstruct_path(current_node)

            if current_node in self.closed_set:
                continue
            self.closed_set.add(current_node)

            # Explore Neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor in self.closed_set:
                    continue
                
                # --- THETA* LOGIC ---
                self.update_vertex(current_node, neighbor)

        print("No path found.")
        return []

    def update_vertex(self, current, neighbor):
        """
        The Core Theta* Logic:
        Try to connect Parent(Current) -> Neighbor (Shortcut).
        If invalid, connect Current -> Neighbor (Standard A*).
        """
        parent = self.graph.nodes[current]["parent"]
        
        # Path 1: The Shortcut (Parent -> Neighbor)
        # We can only shortcut if we have a real parent (not start) AND line of sight
        if parent is not None and parent != current and self.line_of_sight(parent, neighbor):
            
            # Validating Constraints for the LONG segment (Parent -> Neighbor)
            if self.is_valid_move(self.graph.nodes[parent]["parent"], parent, neighbor):
                
                new_g = self.graph.nodes[parent]["g"] + parent.distance(neighbor)
                
                if new_g < self.graph.nodes[neighbor]["g"]:
                    self.graph.nodes[neighbor]["g"] = new_g
                    self.graph.nodes[neighbor]["parent"] = parent # Skip 'current'!
                    self.update_f_score(neighbor)
                    self.push_to_open(neighbor, self.graph.nodes[neighbor]["f"])
                return # If shortcut worked, we are done with this node for now

        # Path 2: Standard A* (Current -> Neighbor)
        # We fall back to this if LOS is blocked OR if the shortcut violated constraints
        if self.is_valid_move(parent, current, neighbor):
            
            new_g = self.graph.nodes[current]["g"] + current.distance(neighbor)
            
            if new_g < self.graph.nodes[neighbor]["g"]:
                self.graph.nodes[neighbor]["g"] = new_g
                self.graph.nodes[neighbor]["parent"] = current
                self.update_f_score(neighbor)
                self.push_to_open(neighbor, self.graph.nodes[neighbor]["f"])

    def is_valid_move(self, grandparent, parent, child):
            """
            Checks Sheet Metal Constraints.
            Allows short segments only if they are collinear (building a flange).
            """
            # 1. Calculate Distances and Angles
            dist = parent.distance(child)
            
            # Calculate angle (default to 0 if no grandparent)
            angle = 0.0
            if grandparent is not None and grandparent != parent:
                angle = self.calculate_bend_angle(grandparent, parent, child)

            # --- RULE 1: MINIMUM FLANGE LENGTH ---
            # We enforce length ONLY if we are making a turn.
            # If we are going straight (angle ~ 0), we allow short segments 
            # because we might be in the middle of "growing" a valid flange.
            
            is_turning = angle > 1.0 # Tolerance for floating point
            is_starting = (grandparent is None) # First step from start
            
            if is_turning:
                # If we turn, the PREVIOUS segment (Grandparent->Parent) must have been long enough.
                # But here we are checking the NEW segment (Parent->Child).
                # Sheet metal rule: You can't put a bend too close to the previous bend.
                # So, strictly speaking, dist(Parent, Child) doesn't need to be > 5.0 YET,
                # but the segment BEFORE the turn must be.
                
                # However, for pathfinding step validation:
                # We usually reject a turn if the *current* segment is tiny, 
                # unless you want to allow "turn, move 1mm, turn" (which is invalid).
                pass 
            
            # CORRECT LOGIC for Grid Expansion:
            # A segment is valid if:
            # A) It meets the length requirement OR
            # B) It is straight (Angle ~ 0) (We are extending a flange) OR
            # C) It is the very first move (We are starting a flange)
            
            if dist < self.minimum_flange_length:
                if is_turning and not is_starting:
                    return False # Cannot turn if the segment is too short!
                # If straight or starting, we ALLOW it (return True at the end)

            if dist > self.maximum_flange_length:
                return False

            # --- RULE 2: BEND ANGLE ---
            # If we ARE turning, the angle must be sufficient.
            if is_turning:
                if angle < self.min_bend_angle:
                    return False
            
            return True

    def line_of_sight(self, p1, p2):
        line = shapely.geometry.LineString([p1, p2])
        # Use a tiny buffer to avoid grazing touches being counted as collisions
        # or simply check if it intersects the INTERIOR of obstacles.
        # Ideally: return not self.environment.blocked_area.intersects(line)
        # If that fails, try:
        return not self.environment.obstacles.intersects(line)

    def calculate_bend_angle(self, p_prev, p_curr, p_next):
        """ Returns deviation from straight line in degrees (0 to 180) """
        v1 = np.array([p_curr.x - p_prev.x, p_curr.y - p_prev.y])
        v2 = np.array([p_next.x - p_curr.x, p_next.y - p_curr.y])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        
        # Dot product angle
        dot = np.dot(v1, v2)
        cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        return angle

    def update_f_score(self, node):
        self.graph.nodes[node]["h"] = node.distance(self.target_point)
        self.graph.nodes[node]["f"] = self.graph.nodes[node]["g"] + self.graph.nodes[node]["h"]

    def push_to_open(self, node, f_score):
        # Tie-breaker counter ensures we never compare Point objects directly if f_scores are equal
        self.counter += 1
        heapq.heappush(self.open_set, (f_score, self.counter, node))

    def reconstruct_path(self, current):
        self.path = []
        while current is not None:
            self.path.append(current)
            parent = self.graph.nodes[current]["parent"]
            # Infinite loop protection
            if parent == current: 
                break
            current = parent
        
        self.path.reverse()
        return self.path

    def check_self_collision(self, current_node, neighbor):
        """
        Returns False if the new segment (current->neighbor) collides 
        with any part of the path history.
        """
        # 1. Define the segment we are trying to add
        new_segment = shapely.geometry.LineString([current_node, neighbor])
        
        # Define a safety margin (e.g., thickness of your sheet metal)
        # If segments get closer than this, it counts as a collision.
        part_thickness = 1.0  # Adjust this to match your material
        
        # 2. Traverse backwards through the lineage
        cursor = current_node
        
        # We skip the first iteration (immediate parent) because we are 
        # connected to it, so intersection is guaranteed at the vertex.
        skip_check = True 
        
        while True:
            parent = self.graph.nodes[cursor]["parent"]
            
            # Stop if we reached the start
            if parent is None:
                break
            
            if skip_check:
                skip_check = False
            else:
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