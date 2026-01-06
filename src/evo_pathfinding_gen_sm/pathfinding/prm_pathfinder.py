import numpy as np
import shapely 
import networkx as nx
import heapq

# Assuming these imports match your project structure
from evo_pathfinding_gen_sm.genetic_algorithm.genetic_code import Genotype
from evo_pathfinding_gen_sm.geometry import Environment

class PRMPathfinder:
    def __init__(self, environment: Environment, genotype: Genotype = None):
        self.environment = environment
        self.path = []
        self.graph = nx.Graph()
        self.start_point = environment.start_point
        self.target_point = environment.target_point
        self.minimum_flange_length = 5.0
        self.maximum_flange_length = 60.0

        # Heap must store tuples: (f_score, node)
        self.open_set = [] 
        self.closed_set = set() # Use a set for O(1) lookups

    # ... [add_nodes and build_roadmap methods remain mostly the same] ...
    # Ensure add_nodes initializes 'bends': 0
    def add_nodes(self, count=1, sampling_mode='random', max_attempts=100, coordinates=None, genotype: Genotype = None):
        added_nodes = []
        attempts = 0
        while len(added_nodes) < count and attempts < max_attempts:
            # ... [Sampling logic same as yours] ...
            # ... inside random/coordinates blocks ...
            
            # Placeholder for point creation to make snippet valid
            if sampling_mode == 'random':
                minx, miny, maxx, maxy = self.environment.free_area.bounds
                point = shapely.geometry.Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            # ...

            if self.environment.free_area.contains(point):
                # Initialize 'bends' count to 0
                self.graph.add_node(point, pos=(point.x, point.y), target=False, start=False, 
                                    g=float('inf'), h=0, f=float('inf'), 
                                    parent=None, bends=0) 
                added_nodes.append(point)
            attempts += 1
        return added_nodes

    def connect_nodes(self, node1, node2):
        if not self.environment.blocked_area.crosses(shapely.geometry.LineString([node1, node2])):
            self.graph.add_edge(node1, node2, weight=node1.distance(node2))

    def build_roadmap(self, sampling_mode='random', connection_mode='nearestn', n=50, node_count=100, radius=30):
        nodes = self.add_nodes(count=node_count, sampling_mode=sampling_mode)
        nodes.append(self.start_point)
        nodes.append(self.target_point)
        
        # Initialize start/target with correct attributes
        self.graph.add_node(self.start_point, pos=(self.start_point.x, self.start_point.y), 
                            target=False, start=True, g=0, h=0, f=0, 
                            parent=self.start_point, bends=0) # Parent of start is start (or None)
        
        self.graph.add_node(self.target_point, pos=(self.target_point.x, self.target_point.y), 
                            target=True, start=False, g=float('inf'), h=0, f=float('inf'), 
                            parent=None, bends=0)

        # ... [Connection logic remains the same] ...
        for node in nodes:
             if connection_mode == 'nearestn':
                distances = [(other_node, node.distance(other_node)) for other_node in nodes if other_node != node]
                distances.sort(key=lambda x: x[1])
                for i in range(min(n, len(distances))):
                    if distances[i][1] <= radius:
                        self.connect_nodes(node, distances[i][0])

    def find_path(self, start_point, target_point):
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

                # Pass parent explicitly to validation to avoid circular lookups
                parent_of_current = self.graph.nodes[current_node]["parent"]
                
                if self.validate_node(parent_of_current, current_node, neighbor):
                    
                    # Calculate new costs
                    new_g, new_bends = self.calculate_g_and_bends(parent_of_current, current_node, neighbor)
                    
                    if new_g < self.graph.nodes[neighbor]["g"]:
                        self.graph.nodes[neighbor]["g"] = new_g
                        self.graph.nodes[neighbor]["h"] = neighbor.distance(target_point)
                        self.graph.nodes[neighbor]["f"] = new_g + self.graph.nodes[neighbor]["h"]
                        self.graph.nodes[neighbor]["parent"] = current_node
                        self.graph.nodes[neighbor]["bends"] = new_bends
                        
                        heapq.heappush(self.open_set, (self.graph.nodes[neighbor]["f"], neighbor))

        # Reconstruct path
        self.path = []
        
        # KEY FIX: Check if path exists before reconstructing
        if self.graph.nodes[target_point]["parent"] is None:
            print("No path found!")
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

    def calculate_g_and_bends(self, parent, current, neighbor):
        # Weights
        length_weight = 1
        bend_weight = 10
        
        # 1. Distance Cost
        dist = current.distance(neighbor)
        
        # 2. Bend Cost
        # Retrieve cumulative bends from current node
        current_bends = self.graph.nodes[current]["bends"]
        is_bend = False
        
        if parent is not None and parent != current:
            # Check if this move creates a new bend
            vec1 = np.array([current.x - parent.x, current.y - parent.y])
            vec2 = np.array([neighbor.x - current.x, neighbor.y - current.y])
            
            # Angle check (simplified: if dot product is not 1, it's a turn)
            # You can use your angle logic here.
            angle_rad = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
            angle_deg = abs(np.degrees(angle_rad)) % 180
            
            # If deviation > small tolerance, count as bend
            if angle_deg > 1.0: 
                is_bend = True
        
        new_bends = current_bends + (1 if is_bend else 0)
        
        # Total G
        new_g = self.graph.nodes[current]["g"] + (dist * length_weight) + ( (1 if is_bend else 0) * bend_weight)
        
        return new_g, new_bends

    def validate_node(self, parent, current, neighbor):
        # 1. Start Node Case: Always valid if within range
        if parent is None or parent == current:
            dist = current.distance(neighbor)
            if dist < self.minimum_flange_length or dist > self.maximum_flange_length:
                return False
            return True

        # 2. Flange Length Check
        dist = current.distance(neighbor)
        if dist < self.minimum_flange_length or dist > self.maximum_flange_length:
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
            
        # If it's effectively straight (0 degrees), it's valid. 
        # If it bends, it must be > 35 degrees.
        if diff_deg > 1.0 and diff_deg < 35:
            return False
            
        return True