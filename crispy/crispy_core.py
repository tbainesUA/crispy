import multiprocessing
import time

import numpy as np
from astropy.io import fits

from .psf import GaussianPSFCube
from .utils import calculate_bin_edges


def initcoef(order, scale, phi, x0=0, y0=0):
    """
    Create a set of coefficients including a rotation matrix plus zeros.
    Parameters
    ----------
    order: int
        The polynomial order of the grid distortion
    scale: float
        The linear separation in pixels of the PSFlets. Default 13.88.
    phi:   float
        The pitch angle of the lenslets.  Default atan(2)
    x0:    float
        x offset to apply to the central pixel. Default 0
    y0:    float
        y offset to apply to the central pixel. Default 0
    Returns
    -------
    coef: list of floats
        A list of length (order+1)*(order+2) to be optimized.
    Notes
    -----
    The list of coefficients has space for a polynomial fit of the
    input order (i.e., for order 3, up to terms like x**3 and x**2*y,
    but not x**3*y).  It is all zeros in the output apart from the
    rotation matrix given by scale and phi.
    """

    try:
        if not order == int(order):
            raise ValueError("Polynomial order must be integer")
        else:
            if order < 1 or order > 5:
                raise ValueError("Polynomial order must be >0, <=5")
    except BaseException:
        raise ValueError("Polynomial order must be integer")

    n = (order + 1) * (order + 2)
    coef = np.zeros((n))

    coef[0] = x0
    coef[1] = scale * np.cos(phi)
    coef[order + 1] = -scale * np.sin(phi)
    coef[n // 2] = y0
    coef[n // 2 + 1] = scale * np.sin(phi)
    coef[n // 2 + order + 1] = scale * np.cos(phi)

    return coef


def transform(x, y, order, coef):
    """
    Apply the coefficients given to transform the coordinates using
    a polynomial.
    Parameters
    ----------
    x:     ndarray
        Rectilinear grid
    y:     ndarray of floats
        Rectilinear grid
    order: int
        Order of the polynomial fit
    coef:  list of floats
        List of the coefficients.  Must match the length required by
        order = (order+1)*(order+2)
    Returns
    -------
    _x:    ndarray
        Transformed coordinates
    _y:    ndarray
        Transformed coordinates
    """

    try:
        if not len(coef) == (order + 1) * (order + 2):
            raise ValueError(
                "Number of coefficients incorrect for polynomial order.")
    except BaseException:
        raise AttributeError("order must be integer, coef should be a list.")

    try:
        if not order == int(order):
            raise ValueError("Polynomial order must be integer")
        else:
            if order < 1 or order > 5:
                raise ValueError("Polynomial order must be >0, <=5")
    except BaseException:
        raise ValueError("Polynomial order must be integer")

    _x = np.zeros(np.asarray(x).shape)
    _y = np.zeros(np.asarray(y).shape)

    i = 0
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            _x += coef[i] * x**ix * y**iy
            i += 1
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            _y += coef[i] * x**ix * y**iy
            i += 1

    return [_x, _y]


def _wrapper_propagate_mono(args):
    ifs, i, lammin, lammax, psf_cube, image = args
    ifs_image = ifs.propagate_mono(lammin, lammax, psf_cube, image_plane=image)
    return (i, ifs_image * (lammax - lammin))

class IFS:
    """Lenslet-based IFS simulator"""
    def __init__(self, lam_ref, R=50, nlens=50, pitch=174e-6, interlace=2, slens=0.5, 
                 npix=1024, pixsize=13e-6, fwhm=2, npixperdlam=2):
        self.lam_ref = lam_ref # reference wavelength
        self.R = R # resolving power
        self.nlens = nlens # number of lenslets
        self.pitch = pitch # lenslet pitch
        self.interlace = interlace # interlacing
        self.slens = slens # lenslet sampling
        self.npix = npix # number of pixels in detector image
        self.pixsize = pixsize # detector pixel size
        self.fwhm = fwhm # fwhm of gaussian kernel
        self.npixperdlam = npixperdlam # number of pixel per spectral resolution element
        
        self.psflets = None
        self.coords = None
        
    @property
    def clocking_angle(self):
        """Lenslet clocking angle"""
        return np.arctan(1/self.interlace)
    
    def propagate_mono(self, lammin, lammax, psfs, image_plane=None, nlam=10):
        
        
        order=3
        npix = 13 # box size
        padwidth = 10
        upsample = 10
        scale = self.pitch / self.pixsize
        angle = self.clocking_angle
        
        # lenslet coordinates
        i_lens, j_lens = np.indices((self.nlens, self.nlens)) - self.nlens // 2 
        
        if image_plane is not None:
            image_plane = np.zeros((self.nlens, self.nlens))
            r = np.hypot(i_lens, j_lens)
            mask = r < self.nlens // 5
            image_plane[mask] = 1
            
            print(image_plane.sum() * np.abs(lammin-lammax))
        
        # detector stuff
        size = self.npix + 2*padwidth
        image = np.zeros((size, size))
        y_det, x_det = np.indices(image.shape)
        
        # wavelengths to integrate over
        wavelengths = np.linspace(lammin, lammax, nlam, endpoint=True)
           
        for wavelength in wavelengths:
            
            # psf models to use
            psf = psfs.interp(wavelength)
            
            # calculate the centroid position on the detector
            dispersion = self.npixperdlam * self.R * np.log(wavelength / self.lam_ref)
            coef = initcoef(order, scale, angle, self.npix // 2 + dispersion, self.npix//2)
            x_cen, y_cen = transform(i_lens, j_lens, order, coef)
            
            x_cen = x_cen.reshape(-1) + padwidth
            y_cen = y_cen.reshape(-1) + padwidth
            
            # print(x_cen, y_cen)
            # image the psf onto the correct position
            for i, (x, y) in enumerate(zip(x_cen, y_cen)):
                if not (x > npix // 2 and x < size - npix // 2 and 
                        y > npix // 2 and y < size - npix // 2):
                    continue
                
                
                if image_plane is not None:
                    a = i_lens.ravel()[i] + image_plane.shape[0] //2
                    b = j_lens.ravel()[i] + image_plane.shape[1]//2
                    
                    val = image_plane[a,b]
                    # print(val)
                    if val == 0:
                        continue
                else:
                    val=1.0
                # print(a,b, val)
                iy1 = int(y) - npix // 2
                iy2 = iy1 + npix
                ix1 = int(x) - npix // 2
                ix2 = ix1 + npix
                
                y_interp = upsample * ((y_det[iy1:iy2, ix1:ix2] - y)  +  npix / 2.)
                x_interp = upsample * ((x_det[iy1:iy2, ix1:ix2] - x)  +  npix / 2.)
                
                # print(type(psf))
                # print(psf.shape)
                psflet = psf.map_psf([y_interp, x_interp])               
               
               
                image[iy1:iy2, ix1:ix2] += val * psflet

        # print(np.sum(image / nlam))
        # return  the padded images
        return image / nlam
    
    def propagate_main(self, lam_bin_centers, image_test=None, lam_bin_edges=None, dlam=None, parallel=True):
        """
        1. generate the padded detector images
        2. multiple spectral cube spaxels with psflets cutouts and combine images into one
        3. remove padding
        4. return image
        """
        
    
        lam_bin_centers = np.array(lam_bin_centers)
        
        nlam = len(lam_bin_centers)
        
        if lam_bin_edges is None:
            if len(lam_bin_centers) > 1:
                lam_bin_edges = calculate_bin_edges(lam_bin_centers)
            else:
                if dlam is None:
                    raise ValueError("No bandwidth specified")
                lam_bin_edges = np.array([lam_bin_centers - dlam / 2, lam_bin_centers + dlam / 2])
                
        else:
            print('Assuming wavelength bin edges were provided') 
        
        # initialize the psf model data cube
        psf_cube = GaussianPSFCube(lam_bin_centers, self.lam_ref, self.fwhm)
        
        # generate image templates
        images = []
        t_start = time.time()
        
        if parallel:
            size = self.npix + 2*10
            shape = (nlam, size,size)
            images = np.zeros(shape)
            ncpu = multiprocessing.cpu_count()
            workers = [(self, i, lam_bin_edges[i], lam_bin_edges[i+1], psf_cube, image_test) 
                       for i in range(nlam)]
            
            with multiprocessing.Pool(ncpu) as pool:
                results = pool.map(_wrapper_propagate_mono, workers)
                
            # store the results in the array and ensure proper order
            iloc= []
            for i in range(nlam):
                images[i] = results[i][1]
                iloc.append(results[i][0])
                
            print(f"Parallel processing return the follow order:\n{iloc}")
                
        else:
            for i in range(nlam):
                lam_min, lam_max = lam_bin_edges[i], lam_bin_edges[i+1]
                image = self.propagate_mono(lam_min, lam_max, psf_cube, image_plane=image_test)
                images.append(image * (lam_max - lam_min))
            
        print(f'Ellapsed time: {time.time() - t_start}')  
        return images
    
    
    
    pass

