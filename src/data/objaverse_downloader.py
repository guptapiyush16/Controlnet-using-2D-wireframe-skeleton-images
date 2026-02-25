import objaverse
import os
import json
from tqdm import tqdm

def download_objaverse_subset(count=100, download_dir="data/raw"):
    """
    Downloads a subset of models from Objaverse.
    """
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Fetching metadata for {count} models...")
    uids = objaverse.load_uids()
    subset_uids = uids[:count]
    
    # Download models
    # Note: objaverse handles caching in ~/.objaverse by default
    processes = 4
    objects = objaverse.load_objects(uids=subset_uids, download_processes=processes)
    
    # Copy/Symlink to our local data directory if needed, or just save the mapping
    mapping_path = os.path.join(download_dir, "uid_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({uid: path for uid, path in objects.items()}, f, indent=4)
    
    print(f"Downloaded {len(objects)} models. Mapping saved to {mapping_path}")
    return objects

if __name__ == "__main__":
    download_objaverse_subset(count=10)
