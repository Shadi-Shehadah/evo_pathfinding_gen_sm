import shapely
import numpy as np
from dataclasses import dataclass
from evo_pathfinding_gen_sm.geometry import Blob

class Genotype:
    def __init__(self,blobs):
        self.blobs = blobs
    
    def get_field_value_vectorized(self, X_mesh, Y_mesh):
        """
        Calculates field intensity for massive grids using NumPy broadcasting.
        X_mesh, Y_mesh are 2D arrays from np.meshgrid.
        """
        total_field = np.zeros_like(X_mesh)
        
        for blob in self.blobs:
            # Calculate squared distance from this blob to ALL points in mesh
            dx = X_mesh - blob.x
            dy = Y_mesh - blob.y
            dist_sq = dx*dx + dy*dy
            
            # Gaussian formula applied to the whole grid at once
            # Avoid division by zero radius just in case
            r_sq = max(blob.radius**2, 1e-9)
            weight = blob.strength * np.exp(-dist_sq / (2 * r_sq))
            
            total_field += weight
            
        return total_field



class Phenotype:
    def __init__(self,bends):
        self.bends = bends
    
    def add_bend(self,bend):
        self.bends.append(bend)

    
class VectorField:
    def __init__(self, phenotype, field_width=2.0):
        """
        field_width (sigma): How 'wide' the influence of the path is.
        """
        self.segments = []
        self.vectors = []
        self.sigma_sq = field_width ** 2
        
        # 1. Pre-calculate segments and their direction vectors
        # Assuming phenotype.bends is a list of Bend objects in order
        if len(phenotype.bends) > 1:
            for i in range(len(phenotype.bends) - 1):
                p1 = phenotype.bends[i].position
                p2 = phenotype.bends[i+1].position
                
                # Create Shapely line for easy distance checks
                segment = shapely.geometry.LineString([p1, p2])
                self.segments.append(segment)
                
                # Calculate normalized direction vector (dx, dy)
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                length = np.hypot(dx, dy)
                
                if length > 0:
                    self.vectors.append(np.array([dx/length, dy/length]))
                else:
                    self.vectors.append(np.array([0.0, 0.0]))

    def get_vector_at(self, x, y):
        """
        Returns the weighted average direction vector (vx, vy) at point (x,y).
        """
        point = shapely.geometry.Point(x, y)
        total_vector = np.array([0.0, 0.0])
        total_weight = 0.0
        
        for i, segment in enumerate(self.segments):
            # Calculate Euclidean distance from Point to Line Segment
            dist = segment.distance(point)
            
            # Gaussian falloff: weight is 1.0 on the line, 0.0 far away
            weight = np.exp(-(dist**2) / (2 * self.sigma_sq))
            
            # Add weighted vector
            total_vector += self.vectors[i] * weight
            total_weight += weight
            
        # Normalize (optional, keeps the "speed" consistent)
        # If you want the field to fade out (speed = 0) far away, remove this block.
        # But usually for pathfinding direction, you just want the angle.
        if total_weight > 0.01: # Avoid division by zero
             return total_vector / total_weight
        
        return np.array([0.0, 0.0])

    def get_vector_field_mesh(self, X_mesh, Y_mesh):
        """
        Vectorized version for the Renderer (calculates whole grid at once).
        """
        # Initialize output grids
        U = np.zeros_like(X_mesh)
        V = np.zeros_like(Y_mesh)
        
        # Loop through segments (easier to loop segments than points for broadcasting)
        for i, segment in enumerate(self.segments):
            # Extract segment coordinates
            x1, y1 = segment.coords[0]
            x2, y2 = segment.coords[1]
            
            # Vector math to find distance from point to segment...
            # (Projecting points onto the line segment clamped 0..1)
            px = x2 - x1
            py = y2 - y1
            norm_sq = px*px + py*py
            
            u = ((X_mesh - x1) * px + (Y_mesh - y1) * py) / float(norm_sq)
            u = np.clip(u, 0, 1)
            
            # Closest point (xc, yc) on the segment
            xc = x1 + u * px
            yc = y1 + u * py
            
            # Distance squared
            dx = X_mesh - xc
            dy = Y_mesh - yc
            dist_sq = dx*dx + dy*dy
            
            # Weight
            weight = np.exp(-dist_sq / (2 * self.sigma_sq))
            
            # Accumulate
            vec = self.vectors[i]
            U += vec[0] * weight
            V += vec[1] * weight
            
        return U, V