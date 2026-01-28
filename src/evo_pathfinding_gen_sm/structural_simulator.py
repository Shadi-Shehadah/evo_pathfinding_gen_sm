
import numpy as np
import shapely
import yaml

with open("../config/config.yaml",'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
material_properties = cfg['material_properties']



class StructuralSimulator:
    def __init__(self, phenotype):


        # 1. Collect Nodes (Start -> Bends -> End)
        # Ensure phenotype.bends is ordered correctly
        self.nodes =[]

        scale_factor = 0.001

        for p in phenotype.nodes:
            scaled_p = shapely.geometry.Point(p.x * scale_factor, p.y * scale_factor)
            self.nodes.append(scaled_p)


        self.num_nodes = len(self.nodes)



        
        # 2. Material & Section Properties
        self.E = material_properties['segment_elastic_modulus']   * 1000000000
        self.A = material_properties['thickness']  *  material_properties['width'] * 0.000001
        self.I = ((material_properties['width'] * material_properties['thickness']**3)  / 12) * 0.001**4 # Moment of Inertia for rectangular section 



    def calculate_max_deflection(self, load_force=100.0):
        """
        Calculates deflection under a test load.
        Returns the displacement magnitude (meters).
        """


        # Degrees of Freedom: 3 per node (u, v, theta)
        num_dof = 3 * self.num_nodes
        K_global = np.zeros((num_dof, num_dof))
        
        # --- ASSEMBLE STIFFNESS MATRIX ---
        for i in range(self.num_nodes - 1):
            n1, n2 = self.nodes[i], self.nodes[i+1]
            dx = n2.x - n1.x
            dy = n2.y - n1.y
            L = np.hypot(dx, dy)
            
            if L < 1e-6: continue # Skip zero-length segments
            
            # Direction Cosines
            c = dx / L
            s = dy / L
            
            # Get Element Stiffness in Global Coordinates
            k_elem = self._get_element_stiffness(L, c, s)
            
            # Add to Global K
            # Map local indices (0..5) to global DOF indices
            idx = [3*i, 3*i+1, 3*i+2, 3*(i+1), 3*(i+1)+1, 3*(i+1)+2]
            
            # Efficiently add the 6x6 matrix to the global matrix using numpy indexing
            # (Looping manually here for clarity, but np.ix_ is faster)
            for r in range(6):
                for col in range(6):
                    K_global[idx[r], idx[col]] += k_elem[r, col]




        # --- APPLY BOUNDARY CONDITIONS ---
        # Fix the Start Node (Node 0): u=0, v=0, theta=0
        fixed_dofs = [0, 1, 2]
        


        # --- APPLY LOAD ---
        # Apply load at the End Node (Last Node)
        # We apply force in Y direction (Vertical Load)
        F = np.zeros(num_dof)
        
        # Index of Y-DOF for the last node
        load_node_idx = 3 * (self.num_nodes - 1) + 1 
        F[load_node_idx] = -load_force # Downward force
        
        # --- SOLVE F = K * u ---
        # Remove fixed rows/cols (Partitioning the matrix)
        free_dofs = [i for i in range(num_dof) if i not in fixed_dofs]
        
        if len(free_dofs) == 0: return 1.0

        try:
            # Extract sub-matrices
            K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
            F_reduced = F[free_dofs]
            
            # Solve linear system
            u_reduced = np.linalg.solve(K_reduced, F_reduced)
            
            # The result is the displacement vector for free DOFs
            # We want the displacement at the load point (End Node)
            # Find where the load node's Y-DOF ended up in the reduced vector
            # It corresponds to the index of 'load_node_idx' inside 'free_dofs'
            if load_node_idx in free_dofs:
                result_idx = free_dofs.index(load_node_idx)
                deflection = abs(u_reduced[result_idx])
                return deflection
            
            else:
                return 0.0 # Should not happen unless end node is fixed
            

        except np.linalg.LinAlgError:
            # Matrix is singular -> Mechanism (unstable structure)
            return float('inf') 
        






    def _get_element_stiffness(self, L, c, s, element_type='beam'):

        """ Returns the 6x6 Element Stiffness Matrix rotated to global coords """
        E, A, I = self.E, self.A, self.I
        


        # 1. Local Stiffness Matrix (Standard Beam Element)
        # Terms for axial and bending stiffness
        a = E * A / L
        b = 12 * E * I / L**3
        d = 6 * E * I / L**2
        e = 4 * E * I / L  # 4EI/L
        f = 2 * E * I / L  # 2EI/L
        
        k_local = np.array([
            [ a,  0,  0, -a,  0,  0],
            [ 0,  b,  d,  0, -b,  d],
            [ 0,  d,  e,  0, -d,  f],
            [-a,  0,  0,  a,  0,  0],
            [ 0, -b, -d,  0,  b, -d],
            [ 0,  d,  f,  0, -d,  e]
        ])
        
        # 2. Rotation Matrix T
        # [ c  s  0  0  0  0 ]
        # [-s  c  0  0  0  0 ]
        # [ 0  0  1  0  0  0 ] ...
        T = np.zeros((6, 6))
        block = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        T[0:3, 0:3] = block
        T[3:6, 3:6] = block
        
        # 3. Rotate: K_global = T_transpose * K_local * T
        return T.T @ k_local @ T