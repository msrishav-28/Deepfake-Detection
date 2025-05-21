import os
import yaml

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config, config_path):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def merge_configs(config1, config2):
    """
    Merge two configurations
    
    Args:
        config1: First configuration dictionary
        config2: Second configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    # Copy config1
    for key, value in config1.items():
        if isinstance(value, dict) and key in config2 and isinstance(config2[key], dict):
            merged[key] = merge_configs(value, config2[key])
        else:
            merged[key] = value
    
    # Copy config2
    for key, value in config2.items():
        if key not in merged:
            merged[key] = value
    
    return merged