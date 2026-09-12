"""Reader for SPARTA shape-definition files (Points/Lines format).

Relocated verbatim (cut/paste, no edits) from
src/rarefied/geometry/process_shape_data.ipynb, cell 1, as part of the
docs/CLEANUP_TODO.md Phase 5 extraction.

NOT the same file format as src/rarefied/io/sparta_runs.py -- that module reads
SPARTA *simulation output* (ITEM tables); this reads SPARTA *shape definitions*
(the Points/Lines geometry format), a different SPARTA feature entirely. No
name collision, but noted here to avoid confusing the two.
"""

def read_sparta_file(filepath):
    """Read a SPARTA format file and extract points and lines."""
    points = {}
    lines = []
    
    with open(filepath, 'r') as f:
        lines_raw = f.readlines()
    
    in_points = False
    in_lines = False
    
    for line in lines_raw:
        line = line.strip()
        if not line:
            continue
        
        if line == "Points":
            in_points = True
            in_lines = False
            continue
        elif line == "Lines":
            in_points = False
            in_lines = True
            continue
        
        if in_points:
            parts = line.split()
            if len(parts) >= 3:
                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                points[idx] = [x, y]
        elif in_lines:
            parts = line.split()
            if len(parts) >= 3:
                line_idx = int(parts[0])
                p1 = int(parts[1])
                p2 = int(parts[2])
                if p1 in points and p2 in points:
                    lines.append((points[p1], points[p2]))
    
    return points, lines
