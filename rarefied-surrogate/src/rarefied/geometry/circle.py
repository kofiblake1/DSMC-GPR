"""Circle-shape generator: analytic circle -> SPARTA format -> regions (+ plot).

Relocated verbatim (cut/paste, no edits) from
src/rarefied/geometry/process_shape_data.ipynb, cell 4, as part of the
docs/CLEANUP_TODO.md Phase 5 extraction.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from rarefied.geometry.mesh import convert_mesh_to_sparta
from rarefied.geometry.sparta_shapes import read_sparta_file
from rarefied.geometry.regions import compute_region_properties, divide_vertices_into_regions, read_region_metadata
from rarefied.geometry.plotting import plot_regions_for_shape, plot_shape

def generate_circle_shape(diameter, num_points=100, output_sparta_path=None, 
                         process_regions=True, max_regions=10, min_points_per_region=10,
                         visualize=True, show_bbox=True, show_fitted_circle=True, show_circles=True,
                         show_normal=True, show_tangent=False, show_area_shading=False,
                         area_plot_samples=80, integration_samples=250):
    """
    Generate a circle shape, convert to SPARTA format, process circle union regions, and optionally visualize.
    
    Parameters:
    - diameter: Circle diameter in model units
    - num_points: Number of points to sample around circle perimeter (default: 100)
    - output_sparta_path: Path to save SPARTA file (default: './generated_circle.txt')
    - process_regions: Whether to compute region metadata (default: True)
    - max_regions: Maximum number of regions (default: 10)
    - min_points_per_region: Minimum points per region (default: 10)
    - visualize: Whether to plot the result (default: True)
    - show_bbox: Show bounding boxes in plot (default: True)
    - show_fitted_circle: Show fitted circles in plot (default: True)
    - show_circles: Show circle union in plot (default: True)
    - show_normal: Show normal vectors in plot (default: True)
    - show_tangent: Show tangent vectors in plot (default: False)
    - show_area_shading: Shade collection area per region for validation (default: False)
    - area_plot_samples: Grid resolution for area shading display (default: 80)
    - integration_samples: Grid resolution for numerical area integration (default: 250)
    
    Returns:
    - dict with keys: 'points', 'lines', 'sparta_path', 'metadata_path' (if processed)
    """
    
    if output_sparta_path is None:
        output_sparta_path = f'generated_circle_d{diameter:.6e}.txt'
    
    # Generate circle points (clockwise order)
    radius = diameter / 2.0
    angles = np.linspace(0, -2*np.pi, num_points, endpoint=False)
    vertices = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles])
    
    # Create edges connecting consecutive vertices
    edges = np.zeros((num_points, 2), dtype=int)
    for i in range(num_points):
        edges[i] = [i + 1, ((i + 1) % num_points) + 1]  # 1-based indexing
    
    # Convert to SPARTA format
    print(f"Generating circle with diameter: {diameter:.6e}")
    print(f"  Number of points: {num_points}")
    print(f"  Radius: {radius:.6e}\n")
    
    success = convert_mesh_to_sparta(vertices, edges, output_sparta_path, 
                                    shape_name='circle', ensure_clockwise=False, target_size=None)
    
    if not success:
        print(f"Error: Failed to convert circle to SPARTA format")
        return None
    
    print(f"✓ Saved SPARTA file: {output_sparta_path}\n")
    
    # Read back the generated SPARTA file
    points, lines = read_sparta_file(output_sparta_path)
    
    result = {
        'points': points,
        'lines': lines,
        'sparta_path': output_sparta_path,
        'vertices': vertices,
        'diameter': diameter
    }
    
    # Process regions if requested
    metadata_path = None
    regions_data = None
    
    if process_regions:
        print("Processing regions with circle union definition...")
        
        # Create a temporary folder for region metadata
        temp_metadata_folder = os.path.join(os.path.dirname(output_sparta_path), 'temp_metadata')
        if not os.path.exists(temp_metadata_folder):
            os.makedirs(temp_metadata_folder)
        
        # Process this single file
        vertices_ordered = np.array([points[i] for i in sorted(points.keys())])
        regions = divide_vertices_into_regions(vertices_ordered, max_regions, min_points_per_region)
        
        # Compute properties for each region
        all_circles = []
        bboxes = []
        normals = []
        tangents = []
        fitted_circles = []
        aoas = []
        union_areas = []
        collection_areas = []
        
        for region in regions:
            circles, bbox, normal, tangent_vec, fitted_circle, aoa, union_area, collection_area = compute_region_properties(
                region_vertices=region,
                object_vertices=vertices_ordered,
                integration_samples=integration_samples
            )
            all_circles.append(circles)
            bboxes.append(bbox)
            normals.append(normal)
            tangents.append(tangent_vec)
            fitted_circles.append(fitted_circle)
            aoas.append(aoa)
            union_areas.append(union_area)
            collection_areas.append(collection_area)
        
        # Save metadata
        shape_name = os.path.splitext(os.path.basename(output_sparta_path))[0]
        metadata_path = os.path.join(temp_metadata_folder, f"{shape_name}_region_metadata.txt")
        
        with open(metadata_path, 'w') as f:
            f.write(f"Region Metadata for: {os.path.basename(output_sparta_path)}\n")
            f.write("=" * 80 + "\n\n")
            f.write("REGION DEFINITION: Circle union (one circle per point)\n")
            f.write("  Each point gets a circle with radius = average distance to neighbors\n\n")
            
            f.write(f"Total vertices: {len(vertices_ordered)}\n")
            f.write(f"Number of regions: {len(regions)}\n")
            f.write(f"Points per region:\n")
            for region_idx, region in enumerate(regions):
                f.write(f"  Region {region_idx}: {len(region)} points\n")
            f.write("\n")
            
            f.write("REGION SUMMARY:\n")
            f.write("Region_Index,Num_Points,BBox_Min_X,BBox_Max_X,BBox_Min_Y,BBox_Max_Y,Normal_X,Normal_Y,Tangent_X,Tangent_Y,Fitted_Circle_X,Fitted_Circle_Y,Fitted_Circle_Radius,Local_AoA_Rad,Union_Area,Collection_Area\n")
            for region_idx, (region, circles, bbox, normal, tangent_vec, fitted_circle, aoa, union_area, collection_area) in enumerate(
                zip(regions, all_circles, bboxes, normals, tangents, fitted_circles, aoas, union_areas, collection_areas)
            ):
                min_x, max_x, min_y, max_y = bbox
                fcx, fcy = fitted_circle['center']
                fr = fitted_circle['radius']
                f.write(f"{region_idx},{len(region)},{min_x:.10e},{max_x:.10e},{min_y:.10e},"
                       f"{max_y:.10e},{normal[0]:.10f},{normal[1]:.10f},{tangent_vec[0]:.10f},{tangent_vec[1]:.10f},"
                       f"{fcx:.10e},{fcy:.10e},{fr:.10e},{aoa:.8e},{union_area:.8e},{collection_area:.8e}\n")
            
            # Write detailed circle data for each region
            f.write("\nDETAILED CIRCLE DATA:\n")
            for region_idx, circles in enumerate(all_circles):
                f.write(f"\nRegion {region_idx} ({len(circles)} circles):\n")
                f.write("Circle_Index,Center_X,Center_Y,Radius\n")
                for circle_idx, circle in enumerate(circles):
                    cx, cy = circle['center']
                    r = circle['radius']
                    f.write(f"{circle_idx},{cx:.10e},{cy:.10e},{r:.10e}\n")
        
        print(f"✓ Saved region metadata: {metadata_path}")
        print(f"  Number of regions: {len(regions)}\n")
        
        # Parse regions for visualization
        regions_data = read_region_metadata(metadata_path)
        result['metadata_path'] = metadata_path
        result['regions'] = regions_data
    
    # Visualize if requested
    if visualize:
        print("Creating visualization...\n")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if regions_data is not None:
            plot_regions_for_shape((points, lines), regions_data,
                                  f"Generated Circle (d={diameter:.6e})",
                                  ax=ax, show_bbox=show_bbox, show_fitted_circle=show_fitted_circle,
                                  show_circles=show_circles, show_normal=show_normal,
                                  show_tangent=show_tangent, show_area_shading=show_area_shading,
                                  area_plot_samples=area_plot_samples, plot_every_n_regions=2)
        else:
            plot_shape(points, lines, f"Generated Circle (d={diameter:.6e})", ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    return result
