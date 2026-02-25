import trimesh
import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
from PIL import Image

def render_wireframe(mesh_path, output_dir, filename_prefix, image_size=(512, 512)):
    """
    Renders a 3D mesh as a 2D RGB image and a 2D wireframe.
    This is a simplified version using trimesh or custom projection logic.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        # Merge all meshes in the scene for simplification
        mesh = mesh.dump(concatenate=True)
        if isinstance(mesh, list):
            # If concatenate=True still returns a list, merge them manually or pick the largest
            mesh = trimesh.util.concatenate(mesh)

    # 1. Generate Wireframe (edges)
    edges = mesh.edges_unique
    vertices = mesh.vertices
    
    # Project vertices to 2D (xy plane)
    # Normalize vertices to [-1, 1]
    vertices = vertices - mesh.centroid
    scale = 1.0 / np.max(np.abs(vertices))
    vertices *= scale
    
    # Map to image coordinates
    v2d = ((vertices[:, :2] + 1) * 0.5 * (image_size[0] - 1)).astype(int)
    
    wireframe_img = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    for edge in edges:
        p1 = tuple(v2d[edge[0]])
        p2 = tuple(v2d[edge[1]])
        cv2.line(wireframe_img, p1, p2, 255, 1)

    # 2. Save Wireframe
    wireframe_path = os.path.join(output_dir, f"{filename_prefix}_wire.png")
    cv2.imwrite(wireframe_path, wireframe_img)
    
    # Placeholder for RGB image (Since we can't easily run a GUI renderer here)
    # Real implementation would use pyrender to get a shaded RGB image
    rgb_path = os.path.join(output_dir, f"{filename_prefix}_rgb.png")
    # For now, let's just save a solid color or the wireframe itself as RGB for structure
    cv2.imwrite(rgb_path, cv2.cvtColor(wireframe_img, cv2.COLOR_GRAY2BGR))

    print(f"Rendered: {wireframe_path}")

if __name__ == "__main__":
    # Batch process downloaded models
    import glob
    
    # Check for downloaded models in the mapping file
    mapping_path = "data/raw/uid_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        
        for uid, mesh_path in mapping.items():
            print(f"Processing {uid}...")
            render_wireframe(mesh_path, "data/processed", uid)
    else:
        # Fallback to searching data/raw for glb files
        glb_files = glob.glob("data/raw/**/*.glb", recursive=True)
        for i, glb_path in enumerate(glb_files):
            render_wireframe(glb_path, "data/processed", f"item_{i}")
