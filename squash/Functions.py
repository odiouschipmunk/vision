"""
Utility functions for squash analysis
"""
import numpy as np

def sum_pixels_in_bbox(frame, bbox):
    """
    Sum all pixel values within a bounding box region
    
    Args:
        frame: Input frame (numpy array)
        bbox: Bounding box [x, y, w, h]
        
    Returns:
        Sum of all pixel values in the region
    """
    x, y, w, h = bbox
    roi = frame[int(y) : int(y + h), int(x) : int(x + w)]
    return np.sum(roi, dtype=np.int64)

def find_match_2d_array(array, x):
    """
    Find if a value exists in the first column of a 2D array
    
    Args:
        array: 2D array to search
        x: Value to find
        
    Returns:
        True if found, False otherwise
    """
    for i in range(len(array)):
        if array[i][0] == x:
            return True
    return False
