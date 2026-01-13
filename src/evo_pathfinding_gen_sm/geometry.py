import shapely
import numpy as np
import networkx as nx



class Environment:
    def __init__(self,area,obstacles,mount1,mount2,width):
        self.width = width
        self.total_area = shapely.box(*area)
        self.obstacles = shapely.geometry.MultiPolygon(obstacles)
        self.mount1 = mount1
        self.mount2 = mount2
        self.blocked_area = shapely.buffer(self.obstacles,self.width/2)
        self.free_area = self.find_free_area()


        self.target_point=shapely.geometry.Point(mount2.plate.coords[0])
        self.start_point=shapely.geometry.Point(mount1.plate.coords[0])

        self.grid = Grid(self.free_area, width, self.target_point, self.start_point)
        

    def add_obstacle(self,obstacle):
        self.obstacles = shapely.geometry.MultiPolygon([*self.obstacles.geoms,obstacle])

        
    
    def find_free_area(self):
        free_area = shapely.difference(self.total_area,self.blocked_area)
        return free_area
    

    def line_of_sight(self,point1,point2):
        line = shapely.geometry.LineString([point1,point2])
        if self.blocked_area.intersects(line):
            return False
        return True
    




class Mount:
    def __init__(self,point1,point2,width):
        self.width = width
        self.plate = self.add_clearance(shapely.geometry.LineString([point1,point2]))
        self.prefer_point1 = True
        
    def add_clearance(self,original: shapely.geometry.LineString):
        return original.offset_curve(-self.width/2)




class Grid:
    def __init__(self, area, node_spacing, fixed_point, added_point):
        self.area = area
        self.node_spacing = node_spacing
        self.fixed_point = fixed_point
        self.added_point = added_point
        self.graph = self.create_graph()

    def create_graph(self):
            graph = nx.Graph()
            
            # Dictionary to store valid nodes: Key=(i, j), Value=Shapely Point
            valid_nodes = {} 
            
            offset_x = (self.fixed_point.x - self.area.bounds[0]) % self.node_spacing
            offset_y = (self.fixed_point.y - self.area.bounds[1]) % self.node_spacing

            x_steps = int((self.area.bounds[2] - self.area.bounds[0]) / self.node_spacing) + 2
            y_steps = int((self.area.bounds[3] - self.area.bounds[1]) / self.node_spacing) + 2

            for i in range(x_steps):
                for j in range(y_steps):
                    node_x = self.area.bounds[0] + (i * self.node_spacing) + offset_x
                    node_y = self.area.bounds[1] + (j * self.node_spacing) + offset_y
                    
                    node = shapely.geometry.Point(node_x, node_y)
                    
                    # Check if the node is inside the valid area
                    if self.area.intersects(node): 
                        graph.add_node(node, pos=(node.x, node.y), target=False, start =False, g=10000, h=0,f=10000, parent = None, is_valid = False)
                        valid_nodes[(i, j)] = node
                                            
                        # 1. Connect to Left Neighbor (i-1, j)
                        if (i - 1, j) in valid_nodes:
                            neighbor = valid_nodes[(i - 1, j)]
                            # Add edge with weight (distance) equal to spacing
                            
                            if not self.area.intersection(shapely.geometry.LineString([node, neighbor])).is_empty: graph.add_edge(node, neighbor, weight=self.node_spacing)
                            
                        # 2. Connect to Bottom Neighbor (i, j-1)
                        if (i, j - 1) in valid_nodes:
                            neighbor = valid_nodes[(i, j - 1)]
                            if not self.area.intersection(shapely.geometry.LineString([node, neighbor])).is_empty: graph.add_edge(node, neighbor, weight=self.node_spacing)
            


            # add the added point to the graph
            graph.add_node(self.added_point, pos=(self.added_point.x, self.added_point.y), g=0, h= 10000, f= 10000, parent = None)

            # mark the target point
            graph.nodes[self.fixed_point]['target'] = True
            # mark the start point
            graph.nodes[self.added_point]['start'] = True
            graph.nodes[self.added_point]['parent'] = self.added_point
            graph.nodes[self.added_point]['is_valid'] = True


            # connect the added point to its nearest neighbors
            for node in graph.nodes:
                if node != self.added_point:
                
                        distance = self.added_point.distance(node)
                        if distance <= self.node_spacing * 1.5:  # connect if within 1.5 times the spacing
                            if not self.area.intersection(shapely.geometry.LineString([self.added_point, node])).is_empty:
                                graph.add_edge(self.added_point, node, weight=distance)
            
            return graph
    
    

class Blob:
    def __init__(self,x,y,radius,strength=1.0):
        self.x = x
        self.y = y
        self.radius = radius
        self.strength = strength



def fillet_linestring(base_linestring,radius):
    pass