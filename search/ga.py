from commons.hw_nats_fast_interface import HW_NATS_FastInterface
from commons.genetics import FastIndividual
from commons.genetics import Genetic, Population
from typing import Iterable, Union, Text
import numpy as np
from collections import OrderedDict

FreeREA_dict = {
    "n": 5,  # tournament size
    "N": 25,  # population size
    "mutation_prob": 1.,  # always mutates
    "recombination_prob": 1.,  # always recombines
    "P_parent1": 0.5,  # fraction of child that comes from parent1 (on average)
    "n_mutations": 1,  # how many loci to mutate at a time
    "loci_prob": None,  # the probability of mutating a given locus (if None, uniform)
}

class GeneticSearch:
    def __init__(self, 
                 searchspace:HW_NATS_FastInterface,
                 genetics_dict:dict=FreeREA_dict,
                 init_population:Union[None, Iterable[FastIndividual]]=None,
                 fitness_weights:Union[None, np.ndarray]=np.array([0.5, 0.5])):
        
        # instantiating a searchspace instance
        self.searchspace = searchspace
        # instatiating the dataset based on searchspace
        self.dataset = self.searchspace.dataset
        # instatiating the device based on searchspace
        self.target_device = self.searchspace.target_device
        # hardware aware scores changes based on whether or not one uses a given target device
        if self.target_device is None:
            self.hw_scores = ["flops", "params"]
        else:
            self.hw_scores = [f"{self.target_device}_energy"]

        # scores used to evaluate the architectures on downstream tasks
        self.classification_scores = ["naswot_score", "logsynflow_score", "skip_score"]
        self.genetics_dict = genetics_dict
        # weights used to combine classification performance with hardware performance.
        self.weights = fitness_weights

        # instantiating a population
        self.population = Population(
            searchspace=self.searchspace, 
            init_population=True if init_population is None else init_population, 
            n_individuals=self.genetics_dict["N"],
            normalization="dynamic"
        )

        # initialize the object taking care of performing genetic operations
        self.genetic_operator = Genetic(
            genome=self.searchspace.all_ops, 
            strategy="comma", # population evolution strategy
            tournament_size=self.genetics_dict["n"], 
            searchspace=self.searchspace
        )

        # preprocess population
        self.preprocess_population()
   
    def normalize_score(self, score_value:float, score_name:Text, type:Text="std")->float:
        """
        Normalize the given score value using a specified normalization type.

        Args:
            score_value (float): The score value to be normalized.
            score_name (Text): The name of the score used for normalization.
            type (Text, optional): The type of normalization to be applied. Defaults to "std".

        Returns:
            float: The normalized score value.

        Raises:
            ValueError: If the specified normalization type is not available.

        Note:
            The available normalization types are:
            - "std": Standard score normalization using mean and standard deviation.
        """
        if type == "std":
            score_mean = self.searchspace.get_score_mean(score_name)
            score_std = self.searchspace.get_score_std(score_name)
            
            return (score_value - score_mean) / score_std
        else:
            raise ValueError(f"Normalization type {type} not available!")

    def fitness_function(self, individual:FastIndividual)->FastIndividual: 
        """
        Directly overwrites the fitness attribute for a given individual.

        Args: 
            individual (FastIndividual): Individual to score.

        # Returns:
        #     FastIndividual: Individual, with fitness field.
        """
        if individual.fitness is None:  # None at initialization only
            scores = np.array([
                self.normalize_score(
                    score_value=self.searchspace.list_to_score(input_list=individual.genotype, 
                                                            score=score), 
                    score_name=score
                )
                for score in self.classification_scores
            ])
            hardware_performance = np.array([
                self.normalize_score(
                    score_value=self.searchspace.list_to_score(input_list=individual.genotype, 
                                                            score=score),
                    score_name=score
                )
                for score in self.hw_scores
            ])
            # individual fitness is a convex combination of multiple scores
            network_score = (np.ones_like(scores) / len(scores)) @ scores
            network_hardware_performance =  (np.ones_like(hardware_performance) / len(hardware_performance)) @ hardware_performance
            
            # in the hardware aware contest performance is in a direct tradeoff with hardware performance
            individual._fitness = np.array([network_score, -network_hardware_performance]) @ self.weights
        
        # return individual

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

    def assign_fitness(self):
        """This function assigns to each invidual a given fitness score."""
        # define a fitness function and compute fitness for each individual
        fitness_function = lambda individual: self.fitness_function(individual=individual)
        self.population.update_fitness(fitness_function=fitness_function)

    def obtain_parents(self, n_parents:int=2): 
        # obtain tournament
        tournament = self.genetic_operator.tournament(population=self.population.individuals)
        # turn tournament into a local population 
        parents = sorted(tournament, key = lambda individual: individual.fitness, reverse=True)[:n_parents]
        return parents

    def solve(self, max_generations:int=100, return_trajectory:bool=False)->Union[FastIndividual, float]: 
        """
        This function performs Regularized Evolutionary Algorithm (REA) with Training-Free metrics. 
        Details on the whole procedure can be found in FreeREA (https://arxiv.org/pdf/2207.05135.pdf).
        
        Args: 
            max_generations (int, optional): TODO - ADD DESCRIPTION. Defaults to 100.
        
        Returns: 
            Union[FastIndividual, float]: Index-0 points to best individual object whereas Index-1 refers to its test 
                                      accuracy.
        """
        
        MAX_GENERATIONS = max_generations
        population, individuals = self.population, self.population.individuals
        bests = []
        history = OrderedDict()

        for gen in range(MAX_GENERATIONS):
            # store the population
            history.update({self.searchspace.list_to_architecture(ind.genotype): ind for ind in population})
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
       
        best_individual = max(history.values(), key=lambda ind: ind._fitness)
        # appending in last position the actual best element
        bests.append(best_individual)

        test_accuracy = self.searchspace.list_to_accuracy(best_individual.genotype)

        if not return_trajectory:
            return (best_individual, test_accuracy)
        else: 
            return (best_individual, test_accuracy, bests)
