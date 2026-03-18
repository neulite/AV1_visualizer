import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

def get_swc_files(directory):
    """Get paths of all .swc files in the specified directory"""
    return glob.glob(os.path.join(directory, '*.swc'))

def read_swc(file_path):
    """Read an SWC file and return the data"""
    data = {}
    soma = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            values = line.strip().split()
            if len(values) == 7:
                idx = int(values[0])
                node_type = int(values[1])
                x = float(values[2])
                y = float(values[3])
                z = float(values[4])
                parent = int(values[6])

                data[idx] = {
                    'type': node_type,
                    'x': x,
                    'y': y,
                    'z': z,
                    'parent': parent
                }
                if node_type == 1 and soma is None:
                    soma = {'x': x, 'y': y, 'z': z}

    # Translate coordinates so that the soma is at the origin
    if soma is not None:
        for idx in data:
            data[idx]['x'] -= soma['x']
            data[idx]['y'] -= soma['y']
            data[idx]['z'] -= soma['z']
    else:
        print("this swc doesn't have soma.")
        exit(1)
    return data
