
import pyevolve
from pyevolve import G1DList, GSimpleGA, Selectors, Mutators, GAllele, Initializators, Crossovers

import numpy as np
import models
import specs
from collections import OrderedDict

chromo_index = None
cache = None

def to_spec(chromo):
    scaled_values = {}
    for val, (spec_name, (low, high)) in zip(chromo, chromo_index.iteritems()):
        scaled_values[spec_name] = low + val*(high-low)
    print scaled_values
    spec_id = specs.create_spec(**scaled_values)
    return specs.get_spec(spec_id)

def make_genome(eval_func, **named_alleles):
    global cache
    cache = {}

    global chromo_index
    chromo_index = OrderedDict()
    for name, scale in named_alleles.iteritems():
        chromo_index[name] = scale

    genome = G1DList.G1DList(len(named_alleles))
    genome.setParams(rangemin=0.0, rangemax=1.0)
    genome.initializator.set(Initializators.G1DListInitializatorReal)
    genome.mutator.set(Mutators.G1DListMutatorRealRange)
    genome.evaluator.set(eval_func)
    genome.crossover.set(Crossovers.G1DListCrossoverUniform)
    return genome

def generate_model(chromo):
    model = models.from_spec(to_spec(chromo))
    model.run()
    return model

def select_for_greater_mean_height(chromo):
    key = tuple(chromo)
    if tuple(chromo) in cache:
        return cache[key]

    model = generate_model(chromo)
    cells = model.cells
    print cells.sum()

    heights = np.zeros(cells.shape[1], int)
    for row in reversed(range(cells.shape[0])):
        heights[np.logical_and((heights == 0), cells[row, :])] = row
    
    value = np.mean(heights)
    cache[key] = value
    print [x for x in chromo], value
    return value

def search_mean_height():
    genome = make_genome(select_for_greater_mean_height,
                         diffusion_constant=(0.01, 10.0), 
                         uptake_rate=(0.01, 3.0))
    ga = GSimpleGA.GSimpleGA(genome)
    ga.selector.set(Selectors.GRouletteWheel)
    ga.setGenerations(20)
    ga.setPopulationSize(80)
    ga.evolve(freq_stats=1)
    print ga.bestIndividual()

if __name__ == '__main__':
    search_mean_height()