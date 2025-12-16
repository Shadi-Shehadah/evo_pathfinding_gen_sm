import matplotlib as plt
import shapely.plotting as plotting
import shapely 

import src.geometry as geometry
import src.genetic_algorithm.genetic_code as genetic_code


def render_environment(env: geometry.Environment):
    plotting.plot_polygon(env.total_area, color='lightgrey', alpha=0.5)
    plotting.plot_polygon(env.obstacles, color='darkgrey', alpha=0.8)
    plotting.plot_polygon(env.free_area, color='lightgreen', alpha=0.3)

def render_phenotype(phenotype: genetic_code.Phenotype):
    pass

def render_genotype(genotype: genetic_code.Genotype):
    pass


# Example usage:
env2 = geometry.Environment(
    area=(0, 0, 10, 10),
    obstacles=[shapely.box(2, 2, 4, 4), shapely.box(6, 6, 8, 8)],
    mount1= geometry.Mount((1, 1), (1, 2), width=0.5),
    mount2= geometry.Mount((9, 9), (9, 8), width=0.5),
    width=0.5
)
render_environment(env2)