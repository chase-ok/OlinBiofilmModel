
import pyevolve
from pyevolve import G1DList, GSimpleGA, Selectors, Mutators, GAllele, Initializators, Crossovers

import numpy as np
import models
import specs

chromo_index = ['boundary_layer',
                'media_penetration']
def to_spec(chromo):
    spec_id = specs.create_spec(**dict(zip(chromo_index, chromo)))
    return specs.get_spec(spec_id)

def make_genome(eval_func, **named_alleles):
    global chromo_index
    chromo_index = []

    alleles = GAllele.GAlleles()
    for name, allele in named_alleles.iteritems():
        chromo_index.append(name)
        alleles.add(allele)

    genome = G1DList.G1DList(len(named_alleles))
    genome.setParams(allele=alleles)
    genome.evaluator.set(eval_func)
    genome.mutator.set(Mutators.G1DListMutatorAllele)
    genome.initializator.set(Initializators.G1DListInitializatorAllele)
    genome.crossover.set(Crossovers.G1DListCrossoverUniform)
    return genome

def generate_model(chromo):
    model = models.from_spec(to_spec(chromo))
    model.run()
    return model

def select_for_greater_mean_height(chromo):
    model = generate_model(chromo)
    cells = model.cells

    heights = np.zeros(cells.shape[1], int)
    for row in reversed(range(cells.shape[0])):
        heights[np.logical_and((heights == 0), cells[row, :])] = row
    
    print [x for x in chromo], np.mean(heights)
    return np.mean(heights)

def search_mean_height():
    genome = make_genome(select_for_greater_mean_height,
                         boundary_layer=GAllele.GAlleleRange(1, 15),
                         media_penetration=GAllele.GAlleleRange(1, 15))
    ga = GSimpleGA.GSimpleGA(genome)
    ga.selector.set(Selectors.GRouletteWheel)
    ga.setGenerations(10)
    ga.setPopulationSize(10)
    ga.evolve(freq_stats=1)
    print ga.bestIndividual()
