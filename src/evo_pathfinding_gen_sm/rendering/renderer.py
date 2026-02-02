import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import shapely.plotting as plotting
import shapely 

from evo_pathfinding_gen_sm.geometry import Environment

from evo_pathfinding_gen_sm.genetic_algorithm.genetic_code import Phenotype, Genotype, VectorField


def render_environment(env: Environment, path: list = None):

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.add_patch(plotting.patch_from_polygon(env.total_area, facecolor='lightgrey', edgecolor='black', alpha=0.5))
    ax.add_patch(plotting.patch_from_polygon(env.obstacles, facecolor='darkgrey', edgecolor='black', alpha=0.8))
    ax.add_patch(plotting.patch_from_polygon(env.free_area, facecolor='lightgreen', edgecolor='black', alpha=0.3))


    if env.grid.graph.number_of_edges() > 0:
        edge_segments = []
        # The graph edges are tuples of Shapely Points: (point_a, point_b)
        for u, v in env.grid.graph.edges():
            edge_segments.append([(u.x, u.y), (v.x, v.y)])
        
        # Create a single collection for all lines
        lc = LineCollection(edge_segments, colors='gray', linewidths=0.5, alpha=0.3)
        ax.add_collection(lc)


    if env.grid.graph.nodes:
        xs = [p.x for p in env.grid.graph.nodes]
        ys = [p.y for p in env.grid.graph.nodes]
        
        ax.scatter(xs, ys, color='red', s=0.1, alpha=0.3, label='Grid Nodes')


    plotting.plot_line(env.mount1.plate, color='blue', linewidth=3)
    plotting.plot_line(env.mount2.plate, color='green', linewidth=3)
    

    #if env.start_point:
     #   ax.plot(env.start_point.x, env.start_point.y, marker='o', color='cyan', markersize=10, label='Start Point')
#
 #   if env.target_point:
  #      ax.plot(env.target_point.x, env.target_point.y, marker='*', color='magenta', markersize=12, label='Target Point')
    
    if path and len(path) > 1:
        path_x = [p.x for p in path]
        path_y = [p.y for p in path]
        ax.plot(path_x, path_y, color='orange', linewidth=2, label='Path')


    
    ax.set_aspect('equal')
    ax.autoscale_view()




    plt.show()

def render_phenotype(phenotype: Phenotype):
    fig, ax = plt.subplots()
    
    # 1. Gather all X and Y coordinates to help auto-scaling later
    all_x = []
    all_y = []

    for bend in phenotype.bends:
        # Plot the node (Bend position)
        ax.plot(bend.position.x, bend.position.y, 'o', color='orange', markersize=8, zorder=3)
        
        # Plot the arrow (Direction)
        # angles='xy', scale_units='xy', scale=1 means: 
        # "An arrow vector of (1,1) will be exactly 1 unit long on the x/y axis"
        #ax.quiver(
        #    bend.position.x, bend.position.y,  # Start X, Y
        #    bend.direction[0], bend.direction[1], # Vector dX, dY
        #    angles='xy', scale_units='xy', scale=1, 
        #    color='blue', width=0.015, zorder=2
        #)

        all_x.append(bend.position.x)
        all_y.append(bend.position.y)
        # Add the tip of the arrow to limits too, so it doesn't get cut off
        all_x.append(bend.position.x + bend.direction[0])
        all_y.append(bend.position.y + bend.direction[1])

    #  Draw the connecting lines
    if len(phenotype.bends) > 1:
        # Create list of points for the line
        line_x = [b.position.x for b in phenotype.bends]
        line_y = [b.position.y for b in phenotype.bends]
        ax.plot(line_x, line_y, color='black', linewidth=2, zorder=1)

    # 3. Force the view to fit the data
    if all_x and all_y:
        buffer = 1.0
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # If there is only ONE point, min == max, so we force a size
        if min_x == max_x:
            min_x -= 1
            max_x += 1
        if min_y == max_y:
            min_y -= 1
            max_y += 1
            
        ax.set_xlim(min_x - buffer, max_x + buffer)
        ax.set_ylim(min_y - buffer, max_y + buffer)

    ax.set_aspect('equal')
    ax.grid(True) # Grid helps visualization
    plt.show()

def render_genotype(genotype: Genotype):
    pass

def render_genotype_field(env: Environment, genotype: Genotype, resolution=150):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- 1. Draw Environment Borders (Changed) ---
    # facecolor='none' ensures the heatmap shows through. 
    # zorder=10 puts the border lines ON TOP of the heatmap.
    ax.add_patch(plotting.patch_from_polygon(
        env.total_area, facecolor='none', edgecolor='black', linewidth=2, zorder=10
    ))
    
    # Draw Obstacles (Opaque, so they cover the field)
    ax.add_patch(plotting.patch_from_polygon(
        env.obstacles, facecolor='darkgrey', edgecolor='black', zorder=11
    ))
    
    # --- 2. Create Sampling Grid ---
    bounds = env.total_area.bounds
    xs = np.linspace(bounds[0], bounds[2], resolution)
    ys = np.linspace(bounds[1], bounds[3], resolution)
    X_mesh, Y_mesh = np.meshgrid(xs, ys)
    
    # --- 3. Calculate Field ---
    Z_field = genotype.get_field_value_vectorized(X_mesh, Y_mesh)
    
    # --- 4. Plot Heatmap (The "Floor" Layer) ---
    cmap = plt.get_cmap('RdBu_r') 
    max_abs_val = np.max(np.abs(Z_field)) if Z_field.size > 0 else 1.0
    norm = mcolors.Normalize(vmin=-max_abs_val, vmax=max_abs_val)
    
    heatmap = ax.imshow(
        Z_field, 
        extent=[bounds[0], bounds[2], bounds[1], bounds[3]], 
        origin='lower',
        cmap=cmap,
        norm=norm,
        alpha=0.8,
        zorder=0  # <--- Crucial: Draws at the very bottom
    )
    
    plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, label='Field Strength')

    # --- 5. Plot Centers & Mounts (Top Layer) ---
    blob_xs = [b.x for b in genotype.blobs]
    blob_ys = [b.y for b in genotype.blobs]
    # zorder=12 ensures these sit on top of everything
    ax.scatter(blob_xs, blob_ys, c='black', marker='x', s=50, zorder=12, label='Sources')

    plotting.plot_line(env.mount1.plate, color='blue', linewidth=3, ax=ax, zorder=12)
    plotting.plot_line(env.mount2.plate, color='green', linewidth=3, ax=ax, zorder=12)
    
    ax.set_aspect('equal')
    ax.set_title("Genotype Potential Field")
    plt.show()


def render_flow_field(env, phenotype, resolution=30):
    """
    Visualizes the continuous flow field generated by a phenotype.
    """
    # 1. Convert Phenotype to Field
    field = VectorField(phenotype.bends, field_width=1.5)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw Background
    ax.add_patch(plotting.patch_from_polygon(env.total_area, facecolor='whitesmoke', edgecolor='black'))
    ax.add_patch(plotting.patch_from_polygon(env.obstacles, facecolor='darkgrey', edgecolor='black'))
    
    # 2. Generate Grid
    bounds = env.total_area.bounds
    xs = np.linspace(bounds[0], bounds[2], resolution)
    ys = np.linspace(bounds[1], bounds[3], resolution)
    X, Y = np.meshgrid(xs, ys)
    
    # 3. Compute Vectors
    U, V = field.get_vector_field_mesh(X, Y)
    
    # 4. Plot Quiver (Arrows)
    # color based on speed (magnitude)
    magnitude = np.sqrt(U**2 + V**2)
    
    # Only draw arrows where there is actually influence (magnitude > 0.1)
    mask = magnitude > 0.1
    
    ax.quiver(
        X[mask], Y[mask], U[mask], V[mask], 
        magnitude[mask], # Color by strength
        cmap='cool', 
        scale=20, 
        width=0.005,
        pivot='mid'
    )
    
    # Draw original path for comparison
    bx = [b.x for b in phenotype.bends]
    by = [b.y for b in phenotype.bends]
    ax.plot(bx, by, 'k--', linewidth=2, alpha=0.5, label='Parent Path')
    
    ax.set_aspect('equal')
    ax.set_title("Flow Field")
    plt.show()

import matplotlib.pyplot as plt
import numpy as np



def render_fitness_graph(generations, title="Fitness Over Generations", fitness_offset=0):
    """
    Plots the best fitness line, the fitness range, and individual candidate points.
    """
    if not generations:
        print("No generations to plot.")
        return

    # 1. Extract Summary Data
    gen_indices = np.arange(len(generations))
    best_fitnesses = []
    worst_fitnesses = []
    
    # 2. Extract Individual Data for Scatter Plot
    all_x = [] # Generation indices
    all_y = [] # Fitness scores

    for i, gen in enumerate(generations):
        scores = [p.fitness + fitness_offset for p in gen.phenotypes]
        
        if scores:
            best_fitnesses.append(max(scores))
            worst_fitnesses.append(min(scores))
            
            # Add every individual score to the scatter lists
            all_x.extend([i] * len(scores))
            all_y.extend(scores)
        else:
            best_fitnesses.append(0)
            worst_fitnesses.append(0)

    # 3. Setup Plot
    plt.figure(figsize=(12, 7)) # Made it slightly wider for detail

    # 4. Plot 'Range' (Shaded Area)
    plt.fill_between(gen_indices, worst_fitnesses, best_fitnesses, 
                     color='gray', alpha=0.2, label='Fitness Range')

    # 5. Plot Individual Candidates (The new feature)
    # 'alpha' makes them transparent so you can see overlapping points
    # 's' is size (small dots)
    # 'zorder' ensures they appear on top of the shading but below the line
    plt.scatter(all_x, all_y, c='blue', alpha=0.3, s=10, label='Candidates', zorder=2)

    # 6. Plot 'Best Fitness' Line
    plt.plot(gen_indices, best_fitnesses, 
             color='black', linewidth=2, label='Best Fitness', zorder=3)

    # 7. Formatting
    plt.title(title, fontsize=16)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.legend(loc='lower right') # Moved legend to avoid covering early data
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.xticks(np.arange(0, len(generations), step=max(1, len(generations)//10)))

    plt.tight_layout()
    plt.show()

