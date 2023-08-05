from pathlib import Path

DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]
DEVICES = ["edgegpu", "eyeriss", "fpga"]

def get_project_root(): 
    """
    Returns project root directory from this script nested in the commons folder.
    """
    return Path(__file__).parent.parent
