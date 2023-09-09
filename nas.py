from src.search import GeneticSearch
from src.hw_nats_fast_interface import HW_NATS_FastInterface
from src.utils import DEVICES
import numpy as np
import argparse

def parse_args()->object: 
    """Args function. 
    Returns:
        (object): args parser
    """
    parser = argparse.ArgumentParser()
    # this selects the dataset to be considered for the search
    parser.add_argument(
        "--dataset", 
        default="cifar10", 
        type=str, 
        help="Dataset to be considered. One in ['cifar10', 'cifar100', 'ImageNet16-120'].s",
        choices=["cifar10", "cifar100", "ImageNet16-120"]
        )
    # this selects the target device to be considered for the search
    parser.add_argument(
        "--device", 
        default="edgegpu", 
        type=str, 
        help="Device to be considered. One in ['edgegpu', 'eyeriss', 'fpga'].",
        choices=["edgegpu", "eyeriss", "fpga"]
        )
    # when this flag is triggered, the search is hardware-agnostic (penalized with FLOPS and params)
    parser.add_argument("--device-agnostic", action="store_true", help="Flag to trigger hardware-agnostic search.")

    parser.add_argument("--n-generations", default=50, type=int, help="Number of generations to let the genetic algorithm run.")
    parser.add_argument("--n-runs", default=30, type=int, help="Number of runs used to compute the average test accuracy.")

    parser.add_argument("--performance-weight", default=0.65, type=float, help="Weight of the performance metric in the fitness function.")
    parser.add_argument("--hardware-weight", default=0.35, type=float, help="Weight of the hardware metric in the fitness function.")

    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()

    dataset = args.dataset
    device = args.device if args.device in DEVICES else None
    n_generations = args.n_generations
    n_runs = args.n_runs
    performance_weight, hardware_weight = args.performance_weight, args.hardware_weight

    if performance_weight + hardware_weight > 1.0 + 1e-6:
        error_msg = f"""
            Performance weight: {performance_weight}, Hardware weight: {hardware_weight} (they sum up to {performance_weight + hardware_weight}).
            The sum of the weights must be less than 1.
        """
        raise ValueError(error_msg)

    # initialize the search space given dataset and device
    searchspace_interface = HW_NATS_FastInterface(device=args.device, dataset=args.dataset)
    search = GeneticSearch(
        searchspace=searchspace_interface, 
        fitness_weights=np.array([performance_weight, hardware_weight])
        )
    # this perform the actual architecture search
    results = search.solve(max_generations=n_generations)

    print(f'{dataset}-{device.upper() if device is not None else device}')
    print(results[0].genotype, results[0].genotype_to_idx["/".join(results[0].genotype)], results[1])
    print()

if __name__=="__main__": 
    main()
