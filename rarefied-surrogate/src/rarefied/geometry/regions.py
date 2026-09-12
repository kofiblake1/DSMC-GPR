"""Circle-union region decomposition for a 2D shape boundary.

Relocated verbatim (cut/paste, no edits) from
src/rarefied/geometry/process_shape_data.ipynb, cell 2, as part of the
docs/CLEANUP_TODO.md Phase 5 extraction.

NOTE (docs/CLEANUP_TODO.md CONFIRM #1, now resolved for both sides): this module's
`read_region_metadata` is a DIFFERENT function from
rarefied.io.sparta_runs.read_region_metadata (moments notebook's tangent-angle
oriented DataFrame reader). This one returns a list of dicts with circle-union
fields (`circles`, `fitted_circle`, `union_area`, `collection_area`, ...) parsed
from the text format `process_sparta_folder`/`generate_circle_shape` in this same
module write. Both were relocated under their original, unchanged names -- across
two different modules there is no runtime collision, and renaming either would
violate the relocation-only rule. Do not `from X import *` both into one namespace.

`compute_region_properties` optionally uses the third-party `circle_fit` package
(imported inline, inside a try/except with a fallback) -- preserved exactly as in
the notebook.
"""

import numpy as np
import os
from matplotlib.path import Path

from rarefied.geometry.sparta_shapes import read_sparta_file

def divide_vertices_into_regions(vertices, max_regions=20, min_points_per_region=5):
    """
    Divide vertices into consecutive regions with constraints:
    - At most max_regions regions
    - At least min_points_per_region points per region
    
    Returns: list of region arrays, where each region is a subset of consecutive vertices
    """
    num_vertices = len(vertices)
    
    # If we can't satisfy constraints, just return single region
    if num_vertices < min_points_per_region or max_regions <= 1:
        return [vertices]
    
    # Choose as many regions as possible while satisfying the minimum points constraint
    num_regions = min(max_regions, num_vertices // min_points_per_region)
    
    if num_regions <= 1:
        return [vertices]
    
    # Split points as evenly as possible across regions
    # Sizes differ by at most 1 point, avoiding a large final region
    base_size = num_vertices // num_regions
    remainder = num_vertices % num_regions
    
    regions = []
    start_idx = 0
    
    for region_idx in range(num_regions):
        region_size = base_size + (1 if region_idx < remainder else 0)
        end_idx = start_idx + region_size
        
        region = vertices[start_idx:end_idx]
        if len(region) > 0:
            regions.append(region)
        start_idx = end_idx
    
    return regions

def compute_region_circle_union(region_vertices, all_vertices):
    """
    Compute a union of circles for a region where each point gets a circle.
    The radius for each point is the average distance to its two neighbors.
    
    Parameters:
    - region_vertices: Nx2 array of vertices in this region
    - all_vertices: Full vertex array (needed to find proper neighbors for edge points)
    
    Returns:
    - circles: list of dicts with keys 'center' (x,y) and 'radius'
    - bounding_box: (min_x, max_x, min_y, max_y) for the union
    """
    circles = []
    n_region = len(region_vertices)
    n_total = len(all_vertices)
    
    # Find starting index of this region in the full vertex array
    region_start_idx = None
    for i in range(n_total - n_region + 1):
        if np.allclose(all_vertices[i], region_vertices[0]) and \
           np.allclose(all_vertices[min(i + n_region - 1, n_total - 1)], region_vertices[-1]):
            region_start_idx = i
            break
    
    if region_start_idx is None:
        # Fallback: use region vertices only
        for local_idx, center in enumerate(region_vertices):
            # Calculate distances to neighbors within region
            if n_region == 1:
                radius = 0.01  # Small default for single point
            elif local_idx == 0:
                dist_next = np.linalg.norm(region_vertices[1] - center)
                radius = dist_next
            elif local_idx == n_region - 1:
                dist_prev = np.linalg.norm(region_vertices[-2] - center)
                radius = dist_prev
            else:
                dist_prev = np.linalg.norm(region_vertices[local_idx - 1] - center)
                dist_next = np.linalg.norm(region_vertices[local_idx + 1] - center)
                radius = (dist_prev + dist_next) / 2.0
            
            circles.append({'center': tuple(center), 'radius': radius})
    else:
        # Use global indices for proper neighbor calculation (handles wrap-around)
        for local_idx in range(n_region):
            global_idx = region_start_idx + local_idx
            center = region_vertices[local_idx]
            
            # Get neighbors with wrap-around
            prev_idx = (global_idx - 1) % n_total
            next_idx = (global_idx + 1) % n_total
            
            dist_prev = np.linalg.norm(all_vertices[prev_idx] - center)
            dist_next = np.linalg.norm(all_vertices[next_idx] - center)
            radius = (dist_prev + dist_next) / 2.0
            
            circles.append({'center': tuple(center), 'radius': radius})
    
    # Compute bounding box for the circle union
    all_radii = np.array([c['radius'] for c in circles])
    all_centers = np.array([c['center'] for c in circles])
    
    min_x = np.min(all_centers[:, 0] - all_radii)
    max_x = np.max(all_centers[:, 0] + all_radii)
    min_y = np.min(all_centers[:, 1] - all_radii)
    max_y = np.max(all_centers[:, 1] + all_radii)
    
    bounding_box = (min_x, max_x, min_y, max_y)
    
    return circles, bounding_box

def estimate_circle_union_area(circles, object_vertices, integration_samples=250):
    """
    Numerically estimate the total collection area for a union of circles.
    Collection area = area of circles NOT inside the object polygon.
    
    Parameters:
    - circles: list of dicts with 'center' and 'radius'
    - object_vertices: Nx2 array of object boundary vertices
    - integration_samples: grid resolution for numerical integration
    
    Returns:
    - union_area: total area covered by circle union
    - collection_area: area of circles outside object (for data collection)
    """
    if not circles:
        return 0.0, 0.0
    
    # Get bounding box for all circles
    all_radii = np.array([c['radius'] for c in circles])
    all_centers = np.array([c['center'] for c in circles])
    
    min_x = np.min(all_centers[:, 0] - all_radii)
    max_x = np.max(all_centers[:, 0] + all_radii)
    min_y = np.min(all_centers[:, 1] - all_radii)
    max_y = np.max(all_centers[:, 1] + all_radii)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    
    # Create integration grid
    x_edges = np.linspace(min_x, max_x, integration_samples + 1)
    y_edges = np.linspace(min_y, max_y, integration_samples + 1)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
    
    xx, yy = np.meshgrid(x_mid, y_mid)
    sample_points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Check which points are inside any circle
    in_circle_union = np.zeros(len(sample_points), dtype=bool)
    for circle in circles:
        cx, cy = circle['center']
        r = circle['radius']
        dist_sq = (sample_points[:, 0] - cx)**2 + (sample_points[:, 1] - cy)**2
        in_circle_union |= (dist_sq <= r**2)
    
    # Check which points are inside object
    polygon_path = Path(object_vertices, closed=True)
    inside_object = polygon_path.contains_points(sample_points)
    
    # Collection area: in circles but NOT in object
    in_collection = in_circle_union & ~inside_object
    
    dx = width / integration_samples
    dy = height / integration_samples
    cell_area = dx * dy
    
    union_area = np.sum(in_circle_union) * cell_area
    collection_area = np.sum(in_collection) * cell_area
    
    return union_area, collection_area

def compute_region_properties(region_vertices, object_vertices=None, integration_samples=250):
    """
    Compute region properties using circle union instead of bounding box.
    
    Returns: (circles, bbox, normal, tangent, fitted_circle, aoa, union_area, collection_area) where:
    - circles: list of dicts {'center': (x, y), 'radius': r} for each point in region
    - bbox: (min_x, max_x, min_y, max_y) bounding box of circle union
    - normal: outward-facing normal vector
    - tangent: tangent vector
    - fitted_circle: single fitted circle for curvature estimation {'center': (cx, cy), 'radius': signed_r}
    - aoa: local angle of attack in radians
    - union_area: total area of circle union
    - collection_area: area available for data collection (circles outside object)
    """
    # Compute circle union for this region
    circles, bbox = compute_region_circle_union(region_vertices, object_vertices)
    
    # Region center for normal computation
    region_center = np.mean(region_vertices, axis=0)
    
    # Outward-facing normal
    if len(region_vertices) >= 2:
        # Use first and last points to estimate tangent
        tangent = region_vertices[-1] - region_vertices[0]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-10:
            tangent = tangent / tangent_norm
            # Normal is perpendicular to tangent
            normal = np.array([-tangent[1], tangent[0]])
            
            # Ensure outward facing by checking against centroid
            center_dir = region_center / (np.linalg.norm(region_center) + 1e-10)
            if np.dot(normal, center_dir) < 0:
                normal = -normal
        else:
            normal = np.array([1.0, 0.0])
    else:
        normal = np.array([1.0, 0.0])
    
    # Tangent from outward normal using right-hand rule in 2D: (-z × n)
    tangent_vec = np.array([normal[1], -normal[0]])
    
    # Fit circle to region vertices for curvature estimation
    try:
        from circle_fit import taubinSVD
        xc, yc, radius, sigma = taubinSVD(region_vertices)
        circle_center = np.array([xc, yc])
    except Exception as e:
        # Fallback: use centroid as center and mean distance as radius
        circle_center = region_center
        radius = np.mean(np.linalg.norm(region_vertices - circle_center, axis=1))
    
    # Compute signed radius
    vec_to_circle = circle_center - region_center
    dot_product = np.dot(normal, vec_to_circle)
    signed_radius = radius if dot_product < 0 else -radius
    
    fitted_circle = {'center': tuple(circle_center), 'radius': signed_radius}
    
    # Compute local angle of attack
    aoa_rad = np.pi - np.arctan2(normal[1], normal[0])
    if aoa_rad > np.pi:
        aoa_rad -= 2 * np.pi
    elif aoa_rad <= -np.pi:
        aoa_rad += 2 * np.pi
    
    # Compute collection area using circle union
    union_area, collection_area = estimate_circle_union_area(
        circles=circles,
        object_vertices=object_vertices,
        integration_samples=integration_samples
    )
    
    return circles, bbox, normal, tangent_vec, fitted_circle, aoa_rad, union_area, collection_area

def process_sparta_folder(sparta_folder, output_metadata_folder=None, max_regions=10, min_points_per_region=10,
                         integration_samples=250):
    """
    Process all SPARTA files in a folder:
    - Divide nodes into regions (at most max_regions, at least min_points_per_region each)
    - Calculate circle unions and normals for each region
    - Save metadata files with region properties
    
    Parameters:
    - sparta_folder: Path to folder containing SPARTA .txt files
    - output_metadata_folder: Path to save metadata (default: sparta_folder/region_metadata)
    - max_regions: Maximum number of regions per shape (default: 10)
    - min_points_per_region: Minimum points per region (default: 10)
    - integration_samples: Numerical integration resolution for area metrics
    """
    if output_metadata_folder is None:
        output_metadata_folder = os.path.join(sparta_folder, 'region_metadata')
    
    if not os.path.exists(output_metadata_folder):
        os.makedirs(output_metadata_folder)
        print(f"Created output folder: {output_metadata_folder}")
    
    # Get all SPARTA files (exclude metadata files)
    sparta_files = sorted([f for f in os.listdir(sparta_folder) 
                          if f.endswith('.txt') and '_metadata' not in f])
    
    print(f"Found {len(sparta_files)} SPARTA files to process")
    print(f"Max regions: {max_regions}, Min points per region: {min_points_per_region}")
    print(f"Integration samples: {integration_samples}")
    print(f"Region definition: Circle union (one circle per point)\n")
    
    for file_idx, sparta_file in enumerate(sparta_files):
        sparta_path = os.path.join(sparta_folder, sparta_file)
        points, lines = read_sparta_file(sparta_path)
        
        if not points:
            print(f"  Skipped {sparta_file}: No points found")
            continue
        
        # Convert points dict to ordered array (object polygon vertices)
        vertices = np.array([points[i] for i in sorted(points.keys())])
        
        # Divide into regions
        regions = divide_vertices_into_regions(vertices, max_regions, min_points_per_region)
        
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
                object_vertices=vertices,
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
        
        # Save metadata file
        base_name = os.path.splitext(sparta_file)[0]
        metadata_filename = f"{base_name}_region_metadata.txt"
        metadata_path = os.path.join(output_metadata_folder, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            f.write(f"Region Metadata for: {sparta_file}\n")
            f.write("=" * 80 + "\n\n")
            f.write("REGION DEFINITION: Circle union (one circle per point)\n")
            f.write("  Each point gets a circle with radius = average distance to neighbors\n\n")
            
            f.write(f"Total vertices: {len(vertices)}\n")
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
        
        if (file_idx + 1) % 10 == 0 or file_idx == len(sparta_files) - 1:
            print(f"  [{file_idx + 1}/{len(sparta_files)}] Processed {sparta_file} ({len(regions)} regions)")
    
    print(f"\n✓ Processing complete!")
    print(f"  Total files processed: {len(sparta_files)}")
    print(f"  Metadata saved to: {output_metadata_folder}")

def read_region_metadata(metadata_filepath):
    """Read region metadata file and extract region properties including circle union data."""
    regions_data = []
    
    with open(metadata_filepath, 'r') as f:
        lines = f.readlines()
    
    # Find start of region summary data
    summary_start_idx = 0
    for idx, line in enumerate(lines):
        if 'Region_Index' in line and 'Num_Points' in line:
            summary_start_idx = idx + 1
            break
    
    # Parse region summary
    idx = summary_start_idx
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.startswith('#') or line.startswith('DETAILED'):
            break
        
        parts = line.split(',')
        if len(parts) >= 16:
            try:
                region_idx = int(parts[0])
                num_points = int(parts[1])
                min_x = float(parts[2])
                max_x = float(parts[3])
                min_y = float(parts[4])
                max_y = float(parts[5])
                normal_x = float(parts[6])
                normal_y = float(parts[7])
                tangent_x = float(parts[8])
                tangent_y = float(parts[9])
                fitted_cx = float(parts[10])
                fitted_cy = float(parts[11])
                fitted_r = float(parts[12])
                aoa = float(parts[13])
                union_area = float(parts[14])
                collection_area = float(parts[15])
                
                regions_data.append({
                    'index': region_idx,
                    'num_points': num_points,
                    'bbox': (min_x, max_x, min_y, max_y),
                    'normal': np.array([normal_x, normal_y]),
                    'tangent': np.array([tangent_x, tangent_y]),
                    'fitted_circle': {'center': (fitted_cx, fitted_cy), 'radius': fitted_r},
                    'aoa': aoa,
                    'union_area': union_area,
                    'collection_area': collection_area,
                    'circles': []  # Will be populated from detailed section
                })
            except (ValueError, IndexError):
                pass
        idx += 1
    
    # Parse detailed circle data
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith('Region ') and '(' in line and 'circles)' in line:
            # Extract region index
            region_idx = int(line.split()[1])
            # Skip header line
            idx += 2
            
            # Read circles for this region
            circles = []
            while idx < len(lines):
                line = lines[idx].strip()
                if not line or line.startswith('Region '):
                    break
                
                parts = line.split(',')
                if len(parts) == 4:
                    try:
                        cx = float(parts[1])
                        cy = float(parts[2])
                        r = float(parts[3])
                        circles.append({'center': (cx, cy), 'radius': r})
                    except ValueError:
                        pass
                idx += 1
            
            # Assign circles to corresponding region
            for region_data in regions_data:
                if region_data['index'] == region_idx:
                    region_data['circles'] = circles
                    break
        else:
            idx += 1
    
    return regions_data
