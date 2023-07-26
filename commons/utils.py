from pathlib import Path

def get_project_root(): 
    """
    Returns project root directory from this script nested in the commons folder.
    """
    return Path(__file__).parent.parent