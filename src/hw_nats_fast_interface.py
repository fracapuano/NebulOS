from typing import Set, Text, List, Tuple, Dict
from itertools import chain
from .utils import get_project_root
import numpy as np
import json

class HW_NATS_FastInterface:    
    def __init__(self, 
                 datapath:str=str(get_project_root()) + "/data/nebuloss.json", 
                 indexpath:str=str(get_project_root()) + "/data/nats_arch_index.json",
                 dataset:str="cifar10", 
                 device:Text="edgegpu", 
                 scores_sample_size:int=1e3):
        
        AVAILABLE_DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]
        AVAILABLE_DEVICES = ["edgegpu", "eyeriss", "fpga"]
        # catch input errors
        if dataset not in AVAILABLE_DATASETS:
            raise ValueError(f"Dataset {dataset} not in {AVAILABLE_DATASETS}!")
        
        if device not in AVAILABLE_DEVICES and device is not None:
            raise ValueError(f"Device {device} not in {AVAILABLE_DEVICES}!")

        # parent init
        with open(datapath, "r") as datafile:
            self._data = {
                int(key): value for key, value in json.load(datafile).items()
            }
        # importing the "/"-architecture <-> index from a json file
        with open(indexpath, "r") as indexfile:
            self._architecture_to_index = json.load(indexfile)

        # store dataset field
        self._dataset = dataset
        self.target_device = device
        # architectures to use to estimate mean and std for scores normalization
        self.random_indices = np.random.choice(len(self), int(scores_sample_size), replace=False)

    def __len__(self)->int:
        """Number of architectures in considered search space."""
        return len(self._data)
    
    def __getitem__(self, idx:int) -> Dict: 
        """Returns (untrained) network corresponding to index `idx`"""
        return self._data[idx]

    def __iter__(self):
        """Iterator method"""
        self.iteration_index = 0
        return self

    def __next__(self):
        if self.iteration_index >= self.__len__():
            raise StopIteration
        # access current element 
        net = self[self.iteration_index]
        # update the iteration index
        self.iteration_index += 1
        return net
    
    @property
    def data(self):
        return self._data

    @property
    def architecture_to_index(self):
        return self._architecture_to_index

    @property 
    def name(self)->Text:
        return "nats"
    
    @property
    def ordered_all_ops(self)->List[Text]:
        """NASTS Bench available operations, ordered (without any precise logic)"""
        return ['skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'avg_pool_3x3']

    @property
    def architecture_len(self)->int: 
        """Returns the number of different operations that uniquevoly define a given architecture"""
        return 6
    
    @property
    def all_ops(self)->Set[Text]:
        """NASTS Bench available operations."""
        return {'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'avg_pool_3x3'}
    
    @property
    def dataset(self)->Text: 
        return self._dataset
    
    @dataset.setter
    def change_dataset(self, new_dataset:Text)->None: 
        """
        Updates the current dataset with a new one. 
        Raises ValueError when new_dataset is not one of ["cifar10", "cifar100", "imagenet16-120"]
        """
        if new_dataset.lower() in self.NATS_datasets: 
            self._dataset = new_dataset
        else: 
            raise ValueError(f"New dataset {new_dataset} not in {self.NATS_datasets}")
    
    def get_score_mean(self, score_name:Text)->float:
        """
        Calculate the mean score value across the dataset for the given score name.

        Args:
            score_name (Text): The name of the score for which to calculate the mean.

        Returns:
            float: The mean score value.

        Note:
            The score values are retrieved from each data point in the dataset and averaged.
        """
        if not hasattr(self, f"mean_{score_name}"):
            # compute the mean on 1000 instances
            mean_score = np.mean([self[i][self.dataset][score_name] for i in self.random_indices])

            # set the mean score accordingly
            setattr(self, f"mean_{score_name}", mean_score)
            self.get_score_mean(score_name=score_name)
        
        return getattr(self, f"mean_{score_name}")

    def get_score_std(self, score_name: Text) -> float:
        """
        Calculate the standard deviation of the score values across the dataset for the given score name.

        Args:
            score_name (Text): The name of the score for which to calculate the standard deviation.

        Returns:
            float: The standard deviation of the score values.

        Note:
            The score values are retrieved from each data point in the dataset, and the standard deviation is calculated.
        """
        if not hasattr(self, f"std_{score_name}"):
            # compute the mean on 1000 instances
            std_score = np.std([self[i][self.dataset][score_name] for i in self.random_indices])

            # set the mean score accordingly
            setattr(self, f"std_{score_name}", std_score)
            self.get_score_std(score_name=score_name)
        
        return getattr(self, f"std_{score_name}")

    def generate_random_samples(self, n_samples:int=10)->Tuple[List[Text], List[int]]:
        """Generate a group of architectures chosen at random"""
        idxs = np.random.choice(self.__len__(), size=n_samples, replace=False)
        cell_structures = [self[i]["architecture_string"] for i in idxs]
        # return tinynets, cell_structures_string and the unique indices of the networks
        return cell_structures, idxs
    
    def list_to_architecture(self, input_list:List[str])->str:
        """
        Reformats genotype as architecture string. 
        This function clearly is specific for this very search space.
        """
        return "|{}|+|{}|{}|+|{}|{}|{}|".format(*input_list)
    
    def architecture_to_list(self, architecture_string:Text)->List[Text]: 
        """Turn architectures string into genotype list

        Args: 
            architecture_string(str): String characterising the cell structure only. 

        Returns: 
            List[str]: List containing the operations in the input cell structure.
                       In a genetic-algorithm setting, this description represents a genotype. 
        """
        # divide the input string into different levels
        subcells = architecture_string.split("+")
        # divide into different nodes to retrieve ops
        ops = chain(*[subcell.split("|")[1:-1] for subcell in subcells])
        
        return list(ops)

    def list_to_accuracy(self, input_list:List[str])->float: 
        """Returns the test accuracy of an input list representing the architecture. 
        This list contains the operations.

        Args:
            input_list (List[str]): List of operations inside the architecture.

        Returns:
            float: Test accuracy (after 200 training epochs).
        """
        # retrieving the index associated to this particular architecture
        arch_index = self.architecture_to_index["/".join(input_list)]
        return self[arch_index][self.dataset]["test_accuracy"]

    def architecture_to_accuracy(self, architecture_string:str)->float:
        """Returns the test accuracy of an architecture string.
        The architecture <-> index map is normalized to be as general as possible, hence some (minor) 
        input processing is needed.

        Args:
            architecture_string (str): Architecture string.

        Returns:
            float: Test accuracy (after 200 training epochs).
        """
        # retrieving the index associated to this particular architecture
        arch_index = self.architecture_to_index["/".join(self.architecture_to_list(architecture_string))]
        return self[arch_index][self.dataset]["test_accuracy"]

    def list_to_score(self, input_list:List[Text], score:Text)->float:
        """Returns the value of `score` of an input list representing the architecture. 
        This list contains the operations.

        Args:
            input_list (List[Text]): List of operations inside the architecture.
            score (Text): Score of interest.

        Returns:
            float: Score value for `input_list`.
        """
        arch_index = self.architecture_to_index["/".join(input_list)]
        return self[arch_index][self.dataset].get(score, None)

    def architecture_to_score(self, architecture_string:Text, score:Text)->float:
        """Returns the value of `score` of an architecture string.
        The architecture <-> index map is normalized to be as general as possible, hence some (minor) 
        input processing is needed.

        Args:
            architecture_string (Text): Architecture string.
            score (Text): Score of interest.

        Returns:
            float: Score value for `architecture_string`.
        """
        # retrieving the index associated to this particular architecture
        arch_index = self.architecture_to_index["/".join(self.architecture_to_list(architecture_string))]
        return self[arch_index][self.dataset].get(score, None)
