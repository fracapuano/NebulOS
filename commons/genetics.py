from typing import Iterable, Callable, Tuple, List, Union, Dict
import numpy as np
from copy import deepcopy as copy
from .utils import *
from itertools import chain
import pandas as pd
from .hw_nats_fast_interface import HW_NATS_FastInterface

class Individual:
    # for typing purposes only 
    pass

class FastIndividual: 
    def __init__(
        self,
        genotype:List[str],
        genotype_to_idx:Dict[str, int],
        index:int, 
        age:int=0):
        
        self._genotype = genotype
        self.index=index
        self.age = age

        self._fitness = None
        self.genotype_to_idx = genotype_to_idx
        
    @property
    def genotype(self): 
        return self._genotype
    
    @property
    def fitness(self): 
        return self._fitness
    
    def update_idx(self):
        self.index = self.genotype_to_idx["/".join(self._genotype)]

    def update_genotype(self, new_genotype:List[str]): 
        """Update current genotype with new one. When doing so, also the network field is updated"""
        self._genotype = new_genotype
        self.update_idx()
    
    def update_fitness(self, metric:Callable, attribute:str="net"): 
        """Update the current value of fitness using provided metric"""
        self._fitness = metric(getattr(self, attribute))
    
    def overwrite_fitness(self, new_fitness:float):
        """Overwrite current value of fitness"""
        if isinstance(new_fitness, float) or isinstance(new_fitness, int): 
            self._fitness = new_fitness
        else: 
            raise ValueError(f"New fitness value ({new_fitness}) is not a number!")
        
class Genetic: 
    def __init__(
        self, 
        genome:Iterable[str], 
        searchspace:HW_NATS_FastInterface):
        strategy:Tuple[str, str]="comma", 
        tournament_size:int=5,
        
        self.genome = set(genome) if not isinstance(genome, set) else genome
        self.strategy = strategy
        self.tournament_size = tournament_size
        self.searchspace = searchspace

    def tournament(self, population:Iterable[Individual]) -> Iterable[Individual]:
        """
        Return tournament, i.e. a random subset of population of size tournament size. 
        Sampling is done without replacement to ensure diversity inside the actual tournament.
        """
        return np.random.choice(a=population, size=self.tournament_size, replace=False).tolist()
    
    def obtain_parents(self, population:Iterable[Individual], n_parents:int=2) -> Iterable[Individual]:
        """Obtain n_parents from population. Parents are defined as the fittest individuals in n_parents tournaments"""
        tournament = self.tournament(population = population)
        # parents are defined as fittest individuals in tournaments
        parents = sorted(tournament, key = lambda individual: individual.fitness, reverse=True)[:n_parents]
        return parents
    
    def mutate(self, 
               individual:Individual, 
               n_loci:int=1, 
               genes_prob:Tuple[None, List[float]]=None) -> Individual: 
        """Applies mutation to a given individual"""
        for _ in range(n_loci): 
            mutant_genotype = copy(individual.genotype)
            # select a locus in the genotype (that is, where mutation will occurr)
            if genes_prob is None:  # uniform probability over all loci
                mutant_locus = np.random.randint(low=0, high=len(mutant_genotype))
            else:  # custom probability distrubution over which locus to mutate
                mutant_locus = np.random.choice(mutant_genotype, p=genes_prob)
            # mapping the locus to the actual gene that will effectively change
            mutant_gene = mutant_genotype[mutant_locus]
            operation, level = mutant_gene.split("~")  # splits the gene into operation and level
            # mutation changes gene, so the current one must be removed from the pool of candidate genes
            mutations = self.genome.difference([operation])
            
            # overwriting the mutant gene with a new one - probability of chosing how to mutate should be selected as well
            mutant_genotype[mutant_locus] = np.random.choice(a=list(mutations)) + f"~{level}"

        mutant_individual = FastIndividual(genotype=None, genotype_to_idx=self.searchspace.architecture_to_index, index=None)
        mutant_individual.update_genotype(mutant_genotype)

        return mutant_individual
    
    def recombine(self, individuals:Iterable[Individual], P_parent1:float=0.5) -> Individual: 
        """Performs recombination of two given `individuals`"""
        if len(individuals) != 2: 
            raise ValueError("Number of individuals cannot be different from 2!")
        
        individual1, individual2 = individuals
        recombinant_genotype = [None for _ in range(len(individual1.genotype))]
        for locus_idx, (gene_1, gene_2) in enumerate(zip(individual1.genotype, individual2.genotype)):
            # chose genes from parent1 according to P_parent1
            recombinant_genotype[locus_idx] = gene_1 if np.random.random() <= P_parent1 else gene_2

        recombinant = FastIndividual(genotype=None, genotype_to_idx=self.searchspace.architecture_to_index, index=None)
        recombinant.update_genotype(list(recombinant_genotype))

        return recombinant

class Population: 
    def __init__(self,
                 space:object,
                 init_population:Union[bool, Iterable]=True,
                 n_individuals:int=20,
                 normalization:str='dynamic'): 
        self.space = space
        if init_population is True:
            self._population = generate_population(searchspace_interface=space, n_individuals=n_individuals)
        else: 
            self._population = init_population
        
        self.oldest = None
        self.worst_n = None
        self.normalization = normalization.lower()
    
    def __iter__(self): 
        for i in self._population: 
            yield i
    
    @property
    def individuals(self):
        return self._population
    
    def update_population(self, new_population:Iterable[Individual]): 
        """Overwrites current population with new one stored in `new_population`"""
        if all([isinstance(el, Individual) for el in new_population]):
            del self._population
            self._population = new_population
        else:
            raise ValueError("new_population is not an Iterable of `Individual` datatype!")

    def fittest_n(self, n:int=1): 
        """Return first `n` individuals based on fitness value"""
        return sorted(self._population, key=lambda individual: individual.fitness, reverse=True)[:n]
    
    def update_ranking(self): 
        """Updates the ranking in the population in light of fitness value"""
        sorted_individuals = sorted(self._population, key=lambda individual: individual.fitness, reverse=True)
        
        # ranking in light of individuals 
        for ranking, individual in enumerate(sorted_individuals):
            individual.update_ranking(new_rank=ranking)

    def update_fitness(self, fitness_function:Callable): 
        """Updates the fitness value of individuals in the population"""
        for individual in self.individuals: 
            individual.overwrite_fitness(fitness_function(individual))
    
    def apply_on_individuals(self, function:Callable)->Union[Iterable, None]: 
        """Applies a function on each individual in the population
        
        Args: 
            function (Callable): function to apply on each individual. Must return an object of class Individual.
        Returns: 
            Union[Iterable, None]: Iterable when inplace=False represents the individuals with function applied.
                                   None represents the output when inplace=True (hence function is applied on the
                                   actual population.
        """
        self._population = [function(individual) for individual in self._population]

    def set_extremes(self, score:str):
        """Set the maximal&minimal value in the population for the score 'score' (must be a class attribute)"""
        if self.normalization == 'dynamic':
            # accessing to the score of each individual
            scores = [getattr(ind, score) for ind in self.individuals]
            min_value = min(scores)
            max_value = max(scores)
        elif self.normalization == 'minmax':
            # extreme_scores is a 2x`number_of_scores`
            min_value, max_value = self.extreme_scores[:, self.scores_dict[score]]
        elif self.normalization == 'standard':
            # extreme_scores is a 2x`number_of_scores`
            mean_value, std_value = self.extreme_scores[:, self.scores_dict[score]]

        if self.normalization in ['minmax', 'dynamic']:
            setattr(self, f"max_{score}", max_value)
            setattr(self, f"min_{score}", min_value)
        else:
            setattr(self, f"mean_{score}", mean_value)
            setattr(self, f"std_{score}", std_value)

    def age(self): 
        """Embeds ageing into the process"""
        def individuals_ageing(individual): 
            individual.age += 1
            return individual

        self.apply_on_individuals(function=individuals_ageing)
    
    def add_to_population(self, new_individuals:Iterable[Individual]): 
        """Add new_individuals to population"""
        # TODO: add a block that if new_individuals are over the current extremes resets those
        self._population = list(chain(self.individuals, new_individuals))
    
    def remove_from_population(self, attribute:str="fitness", n:int=1, ascending:bool=True): 
        """Remove first/last `n` elements from sorted population population in `ascending/descending`
        order based on the value of `attribute`"""
        # TODO: Implement a removal from population strategy that is O(n) (remove min individual) 
        # - currently sorting is obviously not!
        if not all([hasattr(el, attribute) for el in self.individuals]):
            raise ValueError(f"Attribute '{attribute}' is not an attribute of all the individuals!")
        # sort the population based on the value of attribute
        sorted_population = sorted(self.individuals, key=lambda ind: getattr(ind, attribute), reverse=False if ascending else True)
        
        # new population is old population minus the `n` worst individuals with respect to `attribute`
        self.update_population(sorted_population[n:])

    def update_oldest(self, candidate:Individual): 
        """Updates oldest individual in the population"""
        if candidate.age >= self.oldest.age: 
            self.oldest = candidate
        else: 
            pass

    def update_worst_n(self, candidate:Individual, attribute:str="fitness", n:int=2): 
        """Updates worst_n elements in the population"""
        if hasattr(candidate, attribute): 
            if any([getattr(candidate, attribute) < getattr(worst, attribute) for worst in self.worst_n]):
                # candidate is worse than one of the worst individuals
                bad_individuals = self.worst_n + candidate
                # sort in increasing values of fitness
                bad_sorted = sorted(bad_individuals, lambda ind: getattr(ind, attribute))
                self.worst_n = bad_sorted[:n]  # return new worst individuals
    
    def set_oldest(self): 
        """Sets oldest individual in population"""
        self.oldest = max(self.individuals, key=lambda ind: ind.age)
    
    def set_worst_n(self, attribute:str="fitness", n:int=2): 
        """Sets worst n elements based on the value of arbitrary attribute"""
        self.worst_n = sorted(self.individuals, key=lambda ind: getattr(ind, attribute))[:n]
    

def generate_population(searchspace_interface:HW_NATS_FastInterface, n_individuals:int=20)->list: 
    """Generate a population of individuals"""
    # at first generate full cell-structure and unique network indices
    cells, indices = searchspace_interface.generate_random_samples(n_samples=n_individuals)
    
    # mapping strings to list of genes (~genomes)
    genotypes = map(lambda cell: searchspace_interface.architecture_to_list(cell), cells)
    # turn full architecture and cell-structure into genetic population individual
    population = [
        FastIndividual(genotype=genotype, index=index, genototype_to_index=searchspace_interface.architecture_to_index) 
        for genotype, index in zip(genotypes, indices)
    ]
    return population