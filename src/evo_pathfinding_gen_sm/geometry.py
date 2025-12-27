import shapely
import numpy as np



class Environment:
    def __init__(self,area,obstacles,mount1,mount2,width):
        self.width = width
        self.total_area = shapely.box(*area)
        self.obstacles = shapely.geometry.MultiPolygon(obstacles)
        self.mount1 = mount1
        self.mount2 = mount2
        self.blocked_area = shapely.buffer(self.obstacles,self.width/2)
        self.free_area = self.find_free_area()
        self.grid = Grid(self.total_area, node_spacing=0.4, fixed_point=shapely.geometry.Point(mount2.plate.coords[0]),
                          added_point=shapely.geometry.Point(mount1.plate.coords[0]))

    def add_obstacle(self,obstacle):
        self.obstacles = shapely.geometry.MultiPolygon([*self.obstacles.geoms,obstacle])

        
    
    def find_free_area(self):
        free_area = shapely.difference(self.total_area,self.blocked_area)
        return free_area
    
    




class Mount:
    def __init__(self,point1,point2,width):
        self.width = width
        self.plate = self.add_clearance(shapely.geometry.LineString([point1,point2]))
        self.prefer_point1 = True
        
    def add_clearance(self,original: shapely.geometry.LineString):
        return original.offset_curve(-self.width/2)




class Bend:
    def __init__(self,position,direction,previous,following):
        self.position = position
        self.direction = direction
        self.previous = previous
        self.following = following




class Grid:
    def __init__(self, area, node_spacing, fixed_point, added_point):
        self.area = area
        self.node_spacing = node_spacing
        self.fixed_point = fixed_point
        self.added_point = added_point
        self.nodes = self.create_grid_nodes()

    def create_grid_nodes(self):
        nodes_list = [] 
        
        offset_x = (self.fixed_point.x - self.area.bounds[0]) % self.node_spacing
        
        offset_y = (self.fixed_point.y - self.area.bounds[1]) % self.node_spacing

        x_steps = int((self.area.bounds[2] - self.area.bounds[0]) / self.node_spacing) + 2
        y_steps = int((self.area.bounds[3] - self.area.bounds[1]) / self.node_spacing) + 2

        for i in range(x_steps):
            for j in range(y_steps):
                node_x = self.area.bounds[0] + (i * self.node_spacing) + offset_x
                node_y = self.area.bounds[1] + (j * self.node_spacing) + offset_y
                
                node = shapely.geometry.Point(node_x, node_y)
                
                if self.area.contains(node):
                    nodes_list.append(node)
        
        return nodes_list
    

class Blob:
    def __init__(self,x,y,radius,strength=1.0):
        self.x = x
        self.y = y
        self.radius = radius
        self.strength = strength