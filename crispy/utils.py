import numpy as np
from scipy.ndimage import map_coordinates#, rotate



def _round_up_to_odd_integer(value):
    i = np.ceil(value)
    if i % 2 == 0:
        return int(i + 1)
    else:
        return int(i)
    
def nearest(arr, val):
    """Find nearest values in array for a given value, need check if val is out of bounds"""
    if hasattr(val, "__len__"):
        return [nearest(arr, i) for i in val]
    arr = np.asarray(arr)
    idx = np.argmin(np.abs(arr - val))
    closest_val_1 = arr[idx]
    closest_val_2 = arr[idx - 1 if closest_val_1 - val > 0 else idx+1]
    return sorted([idx, idx - 1 if closest_val_1 - val > 0 else idx+1])

def Rotate(image, phi, clip=True, order=3):
    """
    Rotate the input image by phi about its center.  Do not resize the
    image, but pad with zeros.  Function originally from Tim Brandt
    Parameters
    ----------
    image : 2D square array
            Image to rotate
    phi : float
            Rotation angle in radians
    clip :  boolean (optional)
            Clip array by sqrt(2) to remove fill values?  Default True.
    order : integer (optional)
            Order of interpolation when rotating. Default is 1.
    Returns
    -------
    imageout: 2D array
            Rotated image of the same shape as the input image, with zero-padding
    """

    x = np.arange(image.shape[0])
    med_n = np.median(x)
    x -= int(med_n)
    x, y = np.meshgrid(x, x)

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    x = r * np.cos(theta + phi) + med_n
    y = r * np.sin(theta + phi) + med_n

    imageout = map_coordinates(image, [y, x], order=order)

    if clip:
        i = int(imageout.shape[0] * (1. - 1. / np.sqrt(2.)) / 2.)
        imageout = imageout[i:-i, i:-i]

    return imageout