import math

def y_rotation(swc_dict, rad):
    cos_rad = math.cos(rad)
    sin_rad = math.sin(rad)
    
    for point_id, point_data in swc_dict.items():
        x_original = point_data['x']
        z_original = point_data['z']
        
        point_data['x'] = x_original * cos_rad + z_original * sin_rad
        point_data['z'] = z_original * cos_rad - x_original * sin_rad
    
    return swc_dict

def z_rotation(swc_dict, rad):
    cos_rad = math.cos(rad)
    sin_rad = math.sin(rad)
    
    for point_id, point_data in swc_dict.items():
        x_original = point_data['x']
        y_original = point_data['y']
        
        point_data['x'] = x_original * cos_rad - y_original * sin_rad
        point_data['y'] = x_original * sin_rad + y_original * cos_rad
    
    return swc_dict