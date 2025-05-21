# inference/deployment/utils.py
import os
import yaml
from typing import Dict, Any


def load_deployment_config(config_path: str) -> Dict[str, Any]:
    """
    Load deployment configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config