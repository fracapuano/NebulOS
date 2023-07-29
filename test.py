from search.ga import GeneticSearch
from commons.hw_nats_fast_interface import HW_NATS_FastInterface
import numpy as np

for device in ['edgegpu', 'eyeriss', 'fpga', None]:
    for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
        nats_interface = HW_NATS_FastInterface(device=device, dataset=dataset)
        search = GeneticSearch(searchspace=nats_interface, fitness_weights=np.array([.6, 0.1]))

        results = search.solve(max_generations=150)

        print(f'{dataset}-{device.upper() if device is not None else device}')
        print(results[0].genotype, results[0].genotype_to_idx["/".join(results[0].genotype)], results[1])
        print()