"""Readers for SPARTA's multi-timestep velocity-space histogram text output.

Relocated verbatim (cut/paste, no edits) from
src/rarefied/moments/proess_3D_distributions.ipynb, cell 0, as part of the
docs/CLEANUP_TODO.md Phase 4 extraction.
"""

import numpy as np

def read_all_timesteps_histogram_3D(filepath):
    """
    Read 3D histogram data file with multiple timesteps.

    Expected non-comment line formats:
    Header: TimeStep Number-of-bins Total-counts Missing-counts Min-value Max-value
    Bin:    i j k coord_x coord_y coord_z count count_over_total

    Returns:
        dict keyed by timestep, where each value contains header fields and
        arrays for bin indices, coordinates, counts, and normalized counts.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    timesteps_data = {}
    current_timestep = None
    current_header = None
    current_num_bins = None
    bin_idx = []
    bin_coords = []
    number_of_bins = 0
    bin_normalized = []
    normalized = np.array([])

    header_line = True
    idx = 0
    bin_ctr = 0
    while idx < len(lines):
        line = lines[idx].strip()

        # Skip comment lines
        if line.startswith('#') or not line:
            idx += 1
            continue
        
        parts = line.split()

        if header_line:
            timestep = int(parts[0])
            num_bins = int(parts[1])
            total_counts = float(parts[2])
            missing_counts = int(parts[3])
            min_val = float(parts[4])
            max_val = float(parts[5])
            number_of_bins = num_bins**3
            # If we already have data for a previous timestep, save it
            if current_timestep is not None:
                timesteps_data[current_timestep] = {
                    'header': current_header,
                    'bin_coords': bin_coords.copy(),
                    'bin_normalized': normalized.copy()
                }

                coords = np.array(bin_coords)

                T_vals = np.unique(coords[:,0])
                N_vals = np.unique(coords[:,1])
                Z_vals = np.unique(coords[:,2])

                Tdata, Ndata, Zdata = np.meshgrid(
                    T_vals, N_vals, Z_vals, indexing='ij'
                )

                # Tdata = np.array(timesteps_data[current_timestep]['bin_coords'])[:,0]
                # Tdata = Tdata.reshape((num_bins, num_bins, num_bins))
                timesteps_data[current_timestep]['Tdata'] = Tdata
                # Ndata = np.array(timesteps_data[current_timestep]['bin_coords'])[:,1]
                # Ndata = Ndata.reshape((num_bins, num_bins, num_bins))
                timesteps_data[current_timestep]['Ndata'] = Ndata
                # Zdata = np.array(timesteps_data[current_timestep]['bin_coords'])[:,2]
                # Zdata = Zdata.reshape((num_bins, num_bins, num_bins))
                timesteps_data[current_timestep]['Zdata'] = Zdata
                joint_pmf =  timesteps_data[current_timestep]['bin_normalized']
                timesteps_data[current_timestep]['joint_pmf'] = joint_pmf
                # print(f"Loaded timestep {current_timestep} with {len(bin_coords)} bins")

            # Start new timestep
            current_timestep = timestep
            current_header = {
                'TimeStep': timestep,
                'Number_of_bins': num_bins,
                'Total_counts': total_counts,
                'Missing_counts': missing_counts,
                'Min_value': min_val,
                'Max_value': max_val
            }
            bin_coords = []
            bin_normalized = []
            normalized = np.zeros((num_bins, num_bins, num_bins))
            idx += 1
            
            header_line = False
        else:
            bin_idx = (int(parts[0])-1, int(parts[1])-1, int(parts[2])-1)
            coord = (float(parts[3]), float(parts[4]), float(parts[5]))
            count = float(parts[6])
            # normalized = float(parts[7])
            # print("bin_idx:", bin_idx, "coord:", coord, "count:", count, "normalized:", parts[7])
            normalized[bin_idx[0], bin_idx[1], bin_idx[2]] = float(parts[7])
            bin_coords.append(coord)
            # bin_normalized.append(normalized)
            idx += 1
            bin_ctr += 1
            if(bin_ctr == number_of_bins): # hard coding for now
                header_line = True
                bin_ctr = 0

    # Save the last timestep
    if current_timestep is not None:
        timesteps_data[current_timestep] = {
            'header': current_header,
            'bin_coords': bin_coords.copy(),
            'bin_normalized': normalized.copy()
        }
        coords = np.array(bin_coords)

        T_vals = np.unique(coords[:,0])
        N_vals = np.unique(coords[:,1])
        Z_vals = np.unique(coords[:,2])

        Tdata, Ndata, Zdata = np.meshgrid(
            T_vals, N_vals, Z_vals, indexing='ij'
        )

        # Tdata = np.array(timesteps_data[current_timestep]['bin_coords'])[:,0]
        # Tdata = Tdata.reshape((num_bins, num_bins, num_bins))
        timesteps_data[current_timestep]['Tdata'] = Tdata
        # Ndata = np.array(timesteps_data[current_timestep]['bin_coords'])[:,1]
        # Ndata = Ndata.reshape((num_bins, num_bins, num_bins))
        timesteps_data[current_timestep]['Ndata'] = Ndata
        # Zdata = np.array(timesteps_data[current_timestep]['bin_coords'])[:,2]
        # Zdata = Zdata.reshape((num_bins, num_bins, num_bins))
        timesteps_data[current_timestep]['Zdata'] = Zdata
        joint_pmf =  timesteps_data[current_timestep]['bin_normalized']
        timesteps_data[current_timestep]['joint_pmf'] = joint_pmf
    
    return timesteps_data

def read_all_timesteps_histogram_2D(filepath):
    """
    Read 2D histogram data file with multiple timesteps.

    Expected non-comment line formats:
    Header: TimeStep Number-of-bins Total-counts Missing-counts Min-value Max-value
    Bin:    i j coord_x coord_y count count_over_total

    Returns:
        dict keyed by timestep, where each value contains header fields and
        arrays for bin indices, coordinates, counts, and normalized counts.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    timesteps_data = {}
    current_timestep = None
    current_header = None
    current_num_bins = None
    bin_idx = []
    bin_coords = []
    number_of_bins = 0
    bin_normalized = []
    normalized = np.array([])

    header_line = True
    idx = 0
    bin_ctr = 0
    while idx < len(lines):
        line = lines[idx].strip()

        # Skip comment lines
        if line.startswith('#') or not line:
            idx += 1
            continue
        
        parts = line.split()

        if header_line:
            timestep = int(parts[0])
            num_bins = int(parts[1])
            total_counts = float(parts[2])
            missing_counts = int(parts[3])
            min_val = float(parts[4])
            max_val = float(parts[5])
            number_of_bins = num_bins**2
            # If we already have data for a previous timestep, save it
            if current_timestep is not None:
                timesteps_data[current_timestep] = {
                    'header': current_header,
                    'bin_coords': bin_coords.copy(),
                    'bin_normalized': normalized.copy()
                }
                Tdata = np.array(timesteps_data[current_timestep]['bin_coords'])[:,0]
                Tdata = Tdata.reshape((num_bins, num_bins)).T
                timesteps_data[current_timestep]['Tdata'] = Tdata
                Ndata = np.array(timesteps_data[current_timestep]['bin_coords'])[:,1]
                Ndata = Ndata.reshape((num_bins, num_bins)).T
                timesteps_data[current_timestep]['Ndata'] = Ndata
                joint_pmf =  timesteps_data[current_timestep]['bin_normalized'].T
                timesteps_data[current_timestep]['joint_pmf'] = joint_pmf
                # print(f"Loaded timestep {current_timestep} with {len(bin_coords)} bins")

            # Start new timestep
            current_timestep = timestep
            current_header = {
                'TimeStep': timestep,
                'Number_of_bins': num_bins,
                'Total_counts': total_counts,
                'Missing_counts': missing_counts,
                'Min_value': min_val,
                'Max_value': max_val
            }
            bin_coords = []
            bin_normalized = []
            normalized = np.zeros((num_bins, num_bins))
            idx += 1
            
            header_line = False
        else:
            bin_idx = (int(parts[0])-1, int(parts[1])-1)
            coord = (float(parts[2]), float(parts[3]))
            count = float(parts[4])
            # normalized = float(parts[5])
            # print("bin_idx:", bin_idx, "coord:", coord, "count:", count, "normalized:", parts[5])
            normalized[bin_idx[0], bin_idx[1]] = float(parts[5])
            bin_coords.append(coord)
            # bin_normalized.append(normalized)
            idx += 1
            bin_ctr += 1
            if(bin_ctr == number_of_bins): # hard coding for now
                header_line = True
                bin_ctr = 0

    # Save the last timestep
    if current_timestep is not None:
        timesteps_data[current_timestep] = {
            'header': current_header,
            'bin_coords': bin_coords.copy(),
            'bin_normalized': normalized.copy()
        }
        Tdata = np.array(timesteps_data[current_timestep]['bin_coords'])[:,0]
        Tdata = Tdata.reshape((num_bins, num_bins)).T
        timesteps_data[current_timestep]['Tdata'] = Tdata
        Ndata = np.array(timesteps_data[current_timestep]['bin_coords'])[:,1]
        Ndata = Ndata.reshape((num_bins, num_bins)).T
        timesteps_data[current_timestep]['Ndata'] = Ndata
        joint_pmf =  timesteps_data[current_timestep]['bin_normalized'].T
        timesteps_data[current_timestep]['joint_pmf'] = joint_pmf
    
    return timesteps_data
