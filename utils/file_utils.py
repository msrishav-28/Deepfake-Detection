import os
import shutil
import json
import pickle

def ensure_dir(path):
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def copy_file(src, dst):
    """
    Copy file
    
    Args:
        src: Source path
        dst: Destination path
    """
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)

def save_json(data, path):
    """
    Save data as JSON
    
    Args:
        data: Data to save
        path: Path to save
    """
    ensure_dir(os.path.dirname(path))
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    """
    Load data from JSON
    
    Args:
        path: Path to load
        
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)

def save_pickle(data, path):
    """
    Save data as pickle
    
    Args:
        data: Data to save
        path: Path to save
    """
    ensure_dir(os.path.dirname(path))
    
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    """
    Load data from pickle
    
    Args:
        path: Path to load
        
    Returns:
        Loaded data
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def list_files(path, extensions=None):
    """
    List files in directory
    
    Args:
        path: Directory path
        extensions: List of extensions to filter
        
    Returns:
        List of files
    """
    files = []
    
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            
            # Filter by extension
            if extensions:
                ext = os.path.splitext(filename)[1].lower()
                if ext in extensions:
                    files.append(file_path)
            else:
                files.append(file_path)
    
    return files