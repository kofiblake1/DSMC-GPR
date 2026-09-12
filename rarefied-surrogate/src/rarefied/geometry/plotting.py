"""Visualization helpers that generate_circle_shape (circle.py) depends on directly.

Relocated verbatim (cut/paste, no edits) from
src/rarefied/geometry/process_shape_data.ipynb, cells 1 and 2, as part of the
docs/CLEANUP_TODO.md Phase 5 extraction.

These are plotting functions, but unlike the notebooks' other plotting-only
functions (left in the notebook per the README's define-vs-inspect rule),
generate_circle_shape's own default behavior (visualize=True) calls
plot_regions_for_shape/plot_shape directly -- so they had to move too, or
generate_circle_shape could not be relocated without editing its body. Relocating
them unchanged, alongside their own callees (draw_circle_arc,
shade_circle_union_area), is what makes generate_circle_shape's relocation a pure
move rather than a rewrite.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import pdist, squareform

def plot_shape(points, lines, title="Shape", ax=None, target_size=None):
    """Plot a shape from SPARTA format with optional target size circle and max distance line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot edges
    for p1, p2 in lines:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=1.5)
    
    # Plot vertices
    if points:
        vertices = list(points.values())
        vertices_array = np.array(vertices)
        ax.scatter(vertices_array[:, 0], vertices_array[:, 1], color='red', s=10, zorder=5)
        
        # Find and plot the line between the two points with maximum distance
        if len(vertices_array) >= 2:
            pairwise_distances = squareform(pdist(vertices_array))
            max_dist_idx = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)
            p1_max = vertices_array[max_dist_idx[0]]
            p2_max = vertices_array[max_dist_idx[1]]
            max_distance = pairwise_distances[max_dist_idx]
            
            # Draw line between max distance points
            ax.plot([p1_max[0], p2_max[0]], [p1_max[1], p2_max[1]], 'r-', linewidth=2.5, 
                   label=f'Max distance: {max_distance:.6f}', zorder=4)
    
    # Plot target size circle if provided
    if target_size is not None:
        # Find centroid for circle center
        if points:
            vertices = np.array(list(points.values()))
            centroid = np.mean(vertices, axis=0)
            # Draw circle with diameter = target_size
            radius = target_size / 2.0
            circle = plt.Circle(centroid, radius, fill=False, color='green', 
                              linestyle='--', linewidth=2, label=f'Target size: {target_size:.6f}')
            ax.add_patch(circle)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')
    
    return ax

def draw_circle_arc(ax, center, radius, bbox_bounds, draw_bounds, color, linewidth=2, alpha=0.7, linestyle='--'):
    """
    Draw the full circle arc between the endpoints of the bounding box,
    visible within the extended drawing bounds.
    """
    cx, cy = center
    x_min, x_max, y_min, y_max = bbox_bounds
    draw_x_min, draw_x_max, draw_y_min, draw_y_max = draw_bounds
    
    # Generate circle points
    theta = np.linspace(0, 2*np.pi, 500)
    circle_x = cx + radius * np.cos(theta)
    circle_y = cy + radius * np.sin(theta)
    
    # Find indices where circle is within the extended drawing bounds
    in_draw_bounds = (circle_x >= draw_x_min) & (circle_x <= draw_x_max) & \
                     (circle_y >= draw_y_min) & (circle_y <= draw_y_max)
    
    if np.any(in_draw_bounds):
        # Find contiguous segments
        segments = []
        current_segment = []
        
        for i, in_view in enumerate(in_draw_bounds):
            if in_view:
                current_segment.append(i)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
        
        if current_segment:
            segments.append(current_segment)
        
        # Draw each segment
        for segment in segments:
            if len(segment) > 1:
                seg_indices = segment
                ax.plot(circle_x[seg_indices], circle_y[seg_indices], 
                       color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
        
        # Draw the actual bounding box in a lighter shade to show comparison
        rect_actual = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                        fill=False, edgecolor=color, linewidth=1.5, 
                                        alpha=0.4, linestyle=':', zorder=1)
        ax.add_patch(rect_actual)
        
        # Draw the extended drawing bounds as reference
        rect_draw = patches.Rectangle((draw_x_min, draw_y_min), draw_x_max - draw_x_min, draw_y_max - draw_y_min,
                                      fill=False, edgecolor=color, linewidth=0.8, 
                                      alpha=0.2, linestyle=':', zorder=0)
        ax.add_patch(rect_draw)

def shade_circle_union_area(ax, circles, object_vertices, integration_samples=80,
                            inside_color='#ef5350', outside_color='#66bb6a',
                            inside_alpha=0.22, outside_alpha=0.12):
    """
    Shade circle union areas: red for circles inside object, green for collection area.
    """
    if not circles:
        return
    
    # Get bounding box
    all_radii = np.array([c['radius'] for c in circles])
    all_centers = np.array([c['center'] for c in circles])
    
    min_x = np.min(all_centers[:, 0] - all_radii)
    max_x = np.max(all_centers[:, 0] + all_radii)
    min_y = np.min(all_centers[:, 1] - all_radii)
    max_y = np.max(all_centers[:, 1] + all_radii)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width <= 0 or height <= 0:
        return
    
    x_edges = np.linspace(min_x, max_x, integration_samples + 1)
    y_edges = np.linspace(min_y, max_y, integration_samples + 1)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
    
    xx, yy = np.meshgrid(x_mid, y_mid)
    sample_points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Check circle union
    in_circle_union = np.zeros(len(sample_points), dtype=bool)
    for circle in circles:
        cx, cy = circle['center']
        r = circle['radius']
        dist_sq = (sample_points[:, 0] - cx)**2 + (sample_points[:, 1] - cy)**2
        in_circle_union |= (dist_sq <= r**2)
    
    # Check object
    polygon_path = Path(object_vertices, closed=True)
    inside_object = polygon_path.contains_points(sample_points)
    
    in_circle_union = in_circle_union.reshape(len(y_mid), len(x_mid))
    inside_object = inside_object.reshape(len(y_mid), len(x_mid))
    
    # Areas: circles inside object vs. collection area (circles outside object)
    circles_inside_obj = in_circle_union & inside_object
    circles_outside_obj = in_circle_union & ~inside_object
    
    inside_vals = np.ma.masked_where(~circles_inside_obj, np.ones_like(circles_inside_obj, dtype=float))
    outside_vals = np.ma.masked_where(~circles_outside_obj, np.ones_like(circles_outside_obj, dtype=float))
    
    ax.pcolormesh(x_edges, y_edges, outside_vals, shading='flat',
                  cmap=ListedColormap([outside_color]), alpha=outside_alpha, zorder=1)
    ax.pcolormesh(x_edges, y_edges, inside_vals, shading='flat',
                  cmap=ListedColormap([inside_color]), alpha=inside_alpha, zorder=2)

def plot_regions_for_shape(sparta_file, regions_data, title="Shape with Regions", ax=None, 
                           show_bbox=True, show_fitted_circle=True, show_circles=True, show_normal=True, show_tangent=False,
                           show_area_shading=False, area_plot_samples=80, plot_every_n_regions=1,
                           title_fontsize=18, axis_label_fontsize=16, tick_label_fontsize=14,
                           region_label_fontsize=12, area_text_fontsize=10, legend_fontsize=13):
    """
    Plot a shape with its circle union regions.
    
    Parameters:
    - sparta_file: tuple of (points_dict, lines_list)
    - regions_data: list of region dicts with circle union data
    - title: Title for the plot
    - ax: Matplotlib axes (creates new if None)
    - show_bbox: Whether to show bounding boxes (default: True)
    - show_fitted_circle: Whether to show fitted circle for curvature (default: True)
    - show_circles: Whether to show individual circles in the union (default: True)
    - show_normal: Whether to show normal vectors (default: True)
    - show_tangent: Whether to show tangent vectors (default: False)
    - show_area_shading: Shade collection area for validation
    - area_plot_samples: Grid resolution for area shading visualization
    - plot_every_n_regions: Plot only every Nth region (default: 1 = plot all)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    plot_every_n_regions = max(1, int(plot_every_n_regions))
    
    points, lines = sparta_file
    object_vertices = np.array([points[i] for i in sorted(points.keys())]) if points else None
    
    # Plot edges
    for p1, p2 in lines:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=1, alpha=0.5, zorder=3)
    
    # Plot vertices
    if points:
        vertices = np.array(list(points.values()))
        ax.scatter(vertices[:, 0], vertices[:, 1], color='blue', s=15, zorder=6, alpha=0.6)
    
    # Color for each region
    colors = plt.cm.tab10(np.linspace(0, 1, len(regions_data)))
    
    # Calculate axis bounds based on shape vertices
    if points:
        vertices = np.array(list(points.values()))
        x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
        
        # Add padding (10% of shape size)
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.1
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range
        
        visible_bounds = (x_min, x_max, y_min, y_max)
    else:
        visible_bounds = (-0.01, 0.01, -0.01, 0.01)
    
    # Plot regions with circle unions
    for region_idx, region_props in enumerate(regions_data):
        if (region_idx % plot_every_n_regions) != 0:
            continue
        bbox = region_props['bbox']
        normal = region_props['normal']
        tangent = region_props.get('tangent')
        fitted_circle = region_props.get('fitted_circle')
        circles = region_props.get('circles', [])
        aoa = region_props.get('aoa')
        union_area = region_props.get('union_area')
        collection_area = region_props.get('collection_area')
        min_x, max_x, min_y, max_y = bbox

        # Optional area shading for verification
        if show_area_shading and object_vertices is not None and circles:
            shade_circle_union_area(
                ax,
                circles=circles,
                object_vertices=object_vertices,
                integration_samples=area_plot_samples
            )
        
        # Plot individual circles in the union
        if show_circles and circles:
            for circle in circles:
                cx, cy = circle['center']
                r = circle['radius']
                circle_patch = plt.Circle((cx, cy), r, fill=False, edgecolor=colors[region_idx],
                                         linewidth=1.2, alpha=0.6, linestyle='-', zorder=4)
                ax.add_patch(circle_patch)
        
        # Plot bounding box
        if show_bbox:
            rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                     fill=False, edgecolor=colors[region_idx], linewidth=2.0, 
                                     alpha=0.5, linestyle='--', zorder=7)
            ax.add_patch(rect)
        
        # Plot fitted circle for curvature visualization
        if show_fitted_circle and fitted_circle is not None:
            cx, cy = fitted_circle['center']
            r = abs(fitted_circle['radius'])
            bbox_bounds = (min_x, max_x, min_y, max_y)
            
            # Calculate enlarged drawing bounds
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y
            expand_x = bbox_width * 0.5 / 2
            expand_y = bbox_height * 0.5 / 2
            draw_bounds = (min_x - expand_x, max_x + expand_x, 
                          min_y - expand_y, max_y + expand_y)
            
            draw_circle_arc(ax, (cx, cy), r, bbox_bounds, draw_bounds,
                          color=colors[region_idx], linewidth=3, alpha=0.8, linestyle=':')
        
        # Plot outward-facing normal as quiver arrow
        if show_normal:
            box_center_x = (min_x + max_x) / 2
            box_center_y = (min_y + max_y) / 2
            
            shape_size = max(visible_bounds[1] - visible_bounds[0], 
                            visible_bounds[3] - visible_bounds[2])
            scale = shape_size / 25
            
            ax.quiver(box_center_x, box_center_y, normal[0] * scale, normal[1] * scale,
                     angles='xy', scale_units='xy', scale=1, 
                     color=colors[region_idx], alpha=0.9, zorder=8, width=0.0008,
                     headwidth=3, headlength=4)

        # Plot tangent as quiver arrow
        if show_tangent:
            box_center_x = (min_x + max_x) / 2
            box_center_y = (min_y + max_y) / 2
            
            shape_size = max(visible_bounds[1] - visible_bounds[0], 
                            visible_bounds[3] - visible_bounds[2])
            scale = shape_size / 25
            
            tangent_vec = tangent if tangent is not None else np.array([normal[1], -normal[0]])
            ax.quiver(box_center_x, box_center_y, tangent_vec[0] * scale, tangent_vec[1] * scale,
                     angles='xy', scale_units='xy', scale=1,
                     color=colors[region_idx], alpha=0.75, zorder=8, width=0.0008,
                     headwidth=3, headlength=4, linestyle='--')
        
        # Add region label
        box_center_x = (min_x + max_x) / 2
        box_center_y = (min_y + max_y) / 2
        
        curvature_str = ""
        if fitted_circle is not None and fitted_circle['radius'] != 0:
            curvature = 1.0 / fitted_circle['radius']
            curvature_str = f"\nκ={curvature:.3f}"
        
        aoa_str = ""
        if aoa is not None:
            aoa_str = f"\nAoA={aoa:.3f} rad"

        collection_area_str = ""
        if collection_area is not None:
            collection_area_str = f"\nAcol={collection_area:.3e}"
        
        label_text = f'R{region_idx}{curvature_str}{aoa_str}{collection_area_str}'
        ax.text(box_center_x, box_center_y, label_text,
               fontsize=region_label_fontsize, ha='center', va='center', fontweight='bold', color=colors[region_idx], zorder=10,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='none'))

        # Optional detailed area values
        if show_area_shading and union_area is not None and collection_area is not None:
            area_text = f"Aunion={union_area:.2e}\nAcol={collection_area:.2e}"
            ax.text(min_x, max_y, area_text,
                    fontsize=area_text_fontsize, ha='left', va='top', color='black', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    # Set axis limits
    x_min, x_max, y_min, y_max = visible_bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel('X', fontsize=axis_label_fontsize)
    ax.set_ylabel('Y', fontsize=axis_label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    
    # Add legend
    handles = []
    labels = []
    if show_circles:
        handles.append(plt.Line2D([0], [0], color='gray', linewidth=1.2, linestyle='-'))
        labels.append('Circle Union')
    if show_bbox:
        handles.append(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='gray', linewidth=2.0, linestyle='--'))
        labels.append('Bounding Box')
    if show_fitted_circle:
        handles.append(plt.Line2D([0], [0], color='gray', linewidth=3, linestyle=':'))
        labels.append('Fitted Circle')
    if show_area_shading:
        handles.append(patches.Patch(facecolor='#66bb6a', alpha=0.25))
        labels.append('Collection Area')
        handles.append(patches.Patch(facecolor='#ef5350', alpha=0.25))
        labels.append('Circles Inside Object')
    if handles:
        ax.legend(handles, labels, loc='upper right', fontsize=legend_fontsize)
