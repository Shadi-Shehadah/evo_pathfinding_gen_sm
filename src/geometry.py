import shapely
import numpy as np



class Environment:
    def __init__(self,area,obstacles,mount1,mount2,width):
        self.total_area = shapely.box(*area)
        self.obstacles = shapely.geometry.MultiPolygon(obstacles)
        self.mount1 = mount1
        self.mount2 = mount2
        self.free_area = self.find_free_area()
    def add_obstacle(self,obstacle):
        self.obstacles = shapely.geometry.MultiPolygon([*self.obstacles.geoms,obstacle])

        
    
    def find_free_area(self):
        blocked_area = shapely.buffer(self.obstacles,self.width/2)
        free_area = shapely.difference(self.total_area,blocked_area)
        return free_area
    
    




class Mount:
    def __init__(self,point1,point2,width):
        self.plate = self.add_clearance(shapely.geometry.LineString([point1,point2]))
        self.prefer_point1 = True
        self.width = width
    def add_clearance(self,original: shapely.geometry.LineString):
        return original.offset_curve(-self.width/2)




class Bend:
    def __init__(self,position,direction,previous,following):
        self.position = position
        self.direction = direction
        self.previous = previous
        self.following = following





class grid:
    def __init__(self,area,cell_size):
        self.area = area
        self.cell_size = cell_size

