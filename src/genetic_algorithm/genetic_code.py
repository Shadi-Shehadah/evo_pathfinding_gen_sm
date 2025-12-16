import shapely

class Genotype:
    def __init__(self,phenotype):
        self.phenotype = phenotype


class Phenotype:
    def __init__(self,bends):
        self.bends = bends
