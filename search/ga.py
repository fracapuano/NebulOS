from commons.hw_nats_fast_interface import HW_NATS_FastInterface
from commons.genetics import FastIndividual
from commons.genetics import Genetic, Population
from commons.utils import get_project_root
from typing import Callable, Iterable, Union
import numpy as np

FreeREA_dict = {
    "n": 5,  # tournament size
    "N": 25,  # population size
    "mutation_prob": 1.,  # always mutates
    "recombination_prob": 1.,  # always recombines
    "P_parent1": 0.5,  # fraction of child that comes from parent1 on average
    "n_mutations": 1,  # how many loci to mutate
    "loci_prob": None,
}

class GeneticSearch:
    def __init__(self, 
                 searchspace:HW_NATS_FastInterface,
                 dataset:str="cifar10",
                 genetics_dict:dict=FreeREA_dict,
                 target_device:Union[None, str]=None, 
                 init_population:Union[None, Iterable[FastIndividual]]=None,
                 fitness_weights:Union[None, np.ndarray]=np.array([0.5, -0.5])):
        
        self.dataset = dataset
        self.classification_scores = ["naswot_score", "logsynflow_score", "skip_score"]
        # hardware aware scores changes based on whether or not one uses a given target device
        if target_device is None:
            self.hw_scores = ["flops", "params"]
        else:
            self.hw_scores = [f"{target_device}_energy"]
        
        self.genetics_dict = genetics_dict
        # weights used to combine classification performance with hardware performance.
        self.weights = fitness_weights

        # instantiating a searchspace instance
        self.searchspace = searchspace

        # instantiating a population
        self.population = Population(
            space=self.searchspace, 
            init_population=True if init_population is None else init_population, 
            n_individuals=self.genetics_dict["N"],
            normalization="dynamic"
        )

        # initialize the object taking care of performing genetic operations
        self.genetic_operator = Genetic(
            genome=self.searchspace.all_ops, 
            strategy="comma", # population evolution strategy
            tournament_size=self.genetics_dict["n"], 
        )

        # preprocess population
        self.preprocess_population()

    def fitness_function(self, individual:FastIndividual):
        if individual.fitness is None:
            individual._fitness = self.searchspace.list_to_accuracy(
                input_list=individual.genotype
            )

        return individual


    def preprocess_population(self): 
        """
        Applies scoring and fitness function to the whole population. This allows each individual to 
        have the appropriate fields.
        """
        # assign the fitness score
        self.assign_fitness()

    def perform_mutation(
            self,
            individual:FastIndividual,
            )->FastIndividual:
        """Performs mutation with respect to genetic ops parameters"""
        realization = np.random.random()
        if realization <= self.genetics_dict["mutation_prob"]:  # do mutation
            mutant = self.genetic_operator.mutate(
                individual=individual, 
                n_loci=self.genetics_dict["n_mutations"], 
                genes_prob=self.genetics_dict["loci_prob"]
            )
            return mutant
        else:  # don't do mutation
            return individual

    def perform_recombination(
            self, 
            parents:Iterable[FastIndividual],
        )->FastIndividual:
        """Performs recombination with respect to genetic ops parameters"""
        realization = np.random.random()
        if realization <= self.genetics_dict["recombination_prob"]:  # do recombination
            child = self.genetic_operator.recombine(
                individuals=parents, 
                P_parent1=self.genetics_dict["P_parent1"]
            )
            return child
        else:  # don't do recombination - simply return 1st parent
            return parents[0]  

    def compute_fitness(self, individual:FastIndividual): 
        """This function returns the fitness of individuals according to FreeREA's paper"""
        self.searchspace.get_score_mean()

    def assign_fitness(self):
        """This function assigns to each invidual a given fitness score."""
        # define a fitness function and compute fitness for each individual
        fitness_function = lambda individual: self.compute_fitness(individual=individual,
                                                                   population=self.population
                                                                   )
        self.population.update_fitness(fitness_function=fitness_function)

    def get_metrics(self, dataset:str=None)->Iterable[Callable]: 
        """
        This function returns an iterable of functions instantiated relatively to the current
        dataset and a sample batch.
        """
        # boolean switch
        custom_images = False
        if dataset is not None:  # return metrics for custom dataset
            images = load_images(dataset=dataset)
            custom_images = True
        
        images = self.images if not custom_images else images
        # computing the functions with respect to the different available datasets
        get_naswot = lambda individual: score_naswot(
            individual=individual, images=images, lookup_table=self.lookup_table
            )
        get_logsynflow = lambda individual: score_logsynflow(
            individual=individual, images=images, lookup_table=self.lookup_table
            )
        get_skipped = lambda individual: score_skipped(
            individual=individual, lookup_table=self.lookup_table)
        return [get_naswot, get_logsynflow, get_skipped]

    def obtain_parents(self, n_parents:int=2): 
        # obtain tournament
        tournament = self.genetic_operator.tournament(population=self.population.individuals)
        # turn tournament into a local population 
        local_population = Population(space=self.searchspace, init_population=tournament)
        for score in self.score_names:
            local_population.set_extremes(score=score)
        
        # define a fitness function and compute fitness for each individual
        fitness_function = lambda individual: self.compute_fitness(individual=individual,
                                                                   population=local_population
                                                                   )
        local_population.update_fitness(fitness_function=fitness_function)
        parents = sorted(local_population.individuals, key = lambda individual: individual.fitness, reverse=True)[:n_parents]
        return parents

    def solve(self, max_generations:int=100, return_trajectory:bool=False)->Union[FastIndividual, float]: 
        """
        This function performs Regularized Evolutionary Algorithm (REA) with Training-Free metrics. 
        Details on the whole procedure can be found here: https://arxiv.org/pdf/2207.05135.pdf. 
        
        Args: 
            max_generations (int, optional): TODO - ADD DESCRIPTION. Defaults to 100.
        
        Returns: 
            Union[FastIndividual, float]: Index-0 points to best individual object whereas Index-1 refers to its test 
                                      accuracy.
        """
        
        MAX_GENERATIONS = max_generations
        population, individuals = self.population, self.population.individuals
        bests = []
        history = {}

        for gen in range(MAX_GENERATIONS):
            # store the population
            history.update({genotype_to_architecture(ind.genotype): ind for ind in population})
            # save best individual
            bests.append(max(individuals, key=lambda ind: ind.fitness))
            # perform ageing
            population.age()
            # obtain parents
            parents = self.obtain_parents()
            # obtain recombinant child
            child = self.perform_recombination(parents=parents)
            # mutate parents
            mutant1, mutant2 = [self.perform_mutation(parent) for parent in parents]            
            # add mutants and child to population
            population.add_to_population([child, mutant1, mutant2])
            # preprocess the new population - TODO: Implement a only-if-extremes-change strategy
            self.preprocess_population()
            # remove from population worst (from fitness perspective) individuals
            population.remove_from_population(attribute="fitness", n=2)
            # prune from population oldest individual
            population.remove_from_population(attribute="age", ascending=False)
            # overwrite population
            individuals = population.individuals

        # in FreeREA (as in REA) the best individual is defined as the best found so far.
        # this returns the fittest individual in the population of all the architectures ever tried.
        all_individuals = Population(
            space=self.searchspace, 
            init_population=list(history.values()))
        for score in self.score_names:
            all_individuals.set_extremes(score=score)

        compute_fitness = lambda ind: fitness_score(
            ind, 
            population=all_individuals, 
            style=self.population.normalization, 
            weights=self.weights
            )
        
        best_individual = max(all_individuals, key=compute_fitness)
        # appending in last position the actual best element
        bests.append(best_individual)

        # retrieve test accuracy for this individual
        test_metrics = read_test_metrics(dataset=self.dataset)
        test_accuracy = test_metrics[best_individual.index, 1]

        if not return_trajectory:
            return (best_individual, test_accuracy)
        else: 
            return (best_individual, test_accuracy, bests)
