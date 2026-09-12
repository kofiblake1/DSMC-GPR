"""Mesh (.mesh) -> SPARTA shape-format conversion.

Relocated verbatim (cut/paste, no edits) from
src/rarefied/geometry/process_shape_data.ipynb, cell 0, as part of the
docs/CLEANUP_TODO.md Phase 5 extraction.
"""

import numpy as np
import os
from scipy.spatial.distance import pdist, squareform

def read_mesh_file(filepath):
    """Read a .mesh file and extract vertices and edges."""
    vertices = None
    edges = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Read Vertices section
        if line.startswith('Vertices'):
            num_vertices = int(lines[i+1].strip())
            vertices = np.zeros((num_vertices, 2))
            i += 2
            for j in range(num_vertices):
                vertex_line = lines[i+j].strip()
                parts = [float(x) for x in vertex_line.split()]
                vertices[j] = [parts[0], parts[1]]  # Only x, y coordinates
            i += num_vertices
        
        # Read Edges section
        elif line.startswith('Edges'):
            num_edges = int(lines[i+1].strip())
            edges = np.zeros((num_edges, 2), dtype=int)
            i += 2
            for j in range(num_edges):
                edge_line = lines[i+j].strip()
                parts = [int(x) for x in edge_line.split()]
                edges[j] = [parts[0], parts[1]]  # Only first two vertices
            i += num_edges
        
        # Stop at Triangles section
        elif line.startswith('Triangles'):
            break
        else:
            i += 1
    
    return vertices, edges

def compute_winding_order(vertices):
    """
    Compute the winding order of polygon vertices using the shoelace formula.
    Returns: signed area (positive = counter-clockwise, negative = clockwise)
    """
    signed_area = 0.0
    n = len(vertices)
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        signed_area += (v2[0] - v1[0]) * (v2[1] + v1[1])
    return signed_area / 2.0

def ensure_clockwise_order(vertices, edges):
    """
    Sort vertices in clockwise order by angle from centroid.
    Returns: vertices and edges reordered in clockwise order
    """
    # Find centroid
    centroid = np.mean(vertices, axis=0)
    
    # Compute angles from centroid to each vertex
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    
    # Sort by angle in descending order for clockwise ordering
    # (clockwise means decreasing angle when viewed from above)
    sorted_indices = np.argsort(-angles)  # Negative for descending (clockwise)
    
    # Reorder vertices
    vertices_new = vertices[sorted_indices]
    
    # Rebuild edges directly from clockwise order
    # This guarantees connectivity follows vertex order: 1->2->...->N->1
    n_vertices = len(vertices_new)
    edges_new = np.zeros((n_vertices, 2), dtype=int)
    for i in range(n_vertices):
        edges_new[i] = [i + 1, ((i + 1) % n_vertices) + 1]
    
    return vertices_new, edges_new

def scale_shape_to_size(vertices, target_size):
    """
    Scale a shape so that the maximum distance between any two vertices equals target_size.
    
    Parameters:
    - vertices: Nx2 array of vertex coordinates
    - target_size: Target maximum pairwise distance (diameter)
    
    Returns:
    - Scaled vertices
    """
    if len(vertices) < 2:
        return vertices
    
    # Calculate pairwise distances
    pairwise_distances = squareform(pdist(vertices))
    max_distance = np.max(pairwise_distances)
    
    if max_distance > 0:
        scale_factor = target_size / max_distance
        vertices = vertices * scale_factor
    
    return vertices

def extract_boundary_vertices(vertices, edges):
    """
    Extract only boundary vertices (those used in edges).
    
    Parameters:
    - vertices: Nx2 array of all vertex coordinates
    - edges: Mx2 array of edge connectivity (1-based indices)
    
    Returns:
    - boundary_vertices: Kx2 array of boundary vertices only
    - remapped_edges: Mx2 array of edges with indices remapped to boundary vertices
    """
    # Find unique vertex indices used in edges (convert from 1-based to 0-based)
    boundary_indices_set = set()
    for edge in edges:
        boundary_indices_set.add(int(edge[0]) - 1)
        boundary_indices_set.add(int(edge[1]) - 1)
    
    # Sort boundary indices to maintain consistent ordering
    boundary_indices = sorted(list(boundary_indices_set))
    
    # Create mapping from old indices (1-based) to new indices (1-based)
    index_map = {}
    for new_idx, old_idx in enumerate(boundary_indices):
        index_map[old_idx + 1] = new_idx + 1  # Map from 1-based to 1-based
    
    # Extract boundary vertices
    boundary_vertices = vertices[boundary_indices]
    
    # Remap edges to new vertex indices
    remapped_edges = np.zeros_like(edges)
    for i, edge in enumerate(edges):
        old_v1 = int(edge[0])
        old_v2 = int(edge[1])
        remapped_edges[i] = [index_map[old_v1], index_map[old_v2]]
    
    return boundary_vertices, remapped_edges

def convert_mesh_to_sparta(vertices, edges, output_filepath, shape_name=None, ensure_clockwise=True, target_size=None):
    """Convert mesh vertices and edges to SPARTA format and save."""
    if vertices is None or edges is None:
        print(f"Warning: Invalid mesh data for {output_filepath}")
        return False
    
    # Extract only boundary vertices
    vertices, edges = extract_boundary_vertices(vertices, edges)
    
    # Enforce clockwise ordering if requested
    if ensure_clockwise:
        vertices, edges = ensure_clockwise_order(vertices, edges)
    
    # Scale shape if target size is specified
    if target_size is not None:
        vertices = scale_shape_to_size(vertices, target_size)
    
    num_points = len(vertices)
    num_lines = len(edges)
    
    if shape_name is None:
        shape_name = os.path.splitext(os.path.basename(output_filepath))[0]
    
    with open(output_filepath, 'w') as f:
        # Write header
        f.write(f"{shape_name}\n\n")
        f.write(f"{num_points} points\n")
        f.write(f"{num_lines} lines\n\n")
        
        # Write Points section
        f.write("Points\n\n")
        for point_idx, vertex in enumerate(vertices, start=1):
            f.write(f"{point_idx} {vertex[0]:.15e} {vertex[1]:.15e}\n")
        
        # Write Lines section
        f.write("\nLines\n\n")
        for line_idx, edge in enumerate(edges, start=1):
            # Mesh edges use 1-based indexing already, so use them directly
            f.write(f"{line_idx} {int(edge[0])} {int(edge[1])}\n")
    
    return True

def process_mesh_folder(mesh_folder_path, output_folder_path=None, ensure_clockwise=True, target_size=None):
    """
    Read all .mesh files from a folder and convert them to SPARTA format.
    
    Parameters:
    - mesh_folder_path: Path to folder containing meshes subfolder
    - output_folder_path: Path to save converted files (default: mesh_folder/sparta_converted)
    - ensure_clockwise: Whether to verify and enforce clockwise vertex ordering (default: True)
    - target_size: Target maximum pairwise distance for scaling (default: None, no scaling)
    """
    # Construct path to meshes subfolder
    meshes_path = os.path.join(mesh_folder_path, 'meshes')
    
    if not os.path.exists(meshes_path):
        print(f"Error: meshes folder not found at {meshes_path}")
        return
    
    # Create output folder
    if output_folder_path is None:
        output_folder_path = os.path.join(mesh_folder_path, 'sparta_converted')
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output folder: {output_folder_path}")
    
    # Process all .mesh files
    mesh_files = [f for f in os.listdir(meshes_path) if f.endswith('.mesh')]
    mesh_files.sort()
    
    print(f"Found {len(mesh_files)} mesh files to process")
    print(f"Clockwise enforcement: {'ON' if ensure_clockwise else 'OFF'}")
    print(f"Target size scaling: {target_size if target_size is not None else 'OFF'}\n")
    
    for mesh_idx, mesh_file in enumerate(mesh_files):
        mesh_filepath = os.path.join(meshes_path, mesh_file)
        
        # Read mesh file
        vertices, edges = read_mesh_file(mesh_filepath)
        
        if vertices is None or edges is None:
            print(f"  Skipped {mesh_file}: Could not parse vertices/edges")
            continue
        
        # Generate output filename
        base_name = os.path.splitext(mesh_file)[0]
        output_filepath = os.path.join(output_folder_path, f"{base_name}.txt")
        
        # Convert and save
        success = convert_mesh_to_sparta(vertices, edges, output_filepath, 
                                        shape_name=base_name, ensure_clockwise=ensure_clockwise, 
                                        target_size=target_size)
        
        if success:
            if (mesh_idx + 1) % 10 == 0 or mesh_idx == len(mesh_files) - 1:
                print(f"  [{mesh_idx + 1}/{len(mesh_files)}] Processed {base_name}.txt")
        else:
            print(f"  Failed to convert {mesh_file}")
    
    print(f"\n✓ Conversion complete!")
    print(f"  Total files: {len(mesh_files)}")
    print(f"  Output folder: {output_folder_path}")
