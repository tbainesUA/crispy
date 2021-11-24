"""
This module handle the PSF Models used for the IFS simulation.
Primary PSF models include gaussian source and user data that 
could be from lab or other simuatiuon. 

How could we incorparte the use of poppy in this work to construct 
complex PSFs?

Is it possible to add diffraction affects to the PSF models so that 
different types of PSF can be used? Charis shows diffraction spikes
from the square lenslets: 

https://www.spiedigitallibrary.org/journals/Journal-of-Astronomical-Telescopes-Instruments-and-Systems/volume-3/issue-04/048002/Data-reduction-pipeline-for-the-CHARIS-integral-field-spectrograph-I/10.1117/1.JATIS.3.4.048002.full?SSO=1
"""


import copy

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from scipy.ndimage import map_coordinates

from .utils import _round_up_to_odd_integer, nearest


def get_gausssian_psf_cube(wavelengths, lam_fwhm, fwhm, npix=13, oversample=10):
    """Testing fucntion for """
    size = _round_up_to_odd_integer(oversample * (npix + 1))
    sigma = fwhm * gaussian_fwhm_to_sigma * oversample * wavelengths / lam_fwhm
    gaussian_psfs = [Gaussian2DKernel(sig, x_size=size).array for sig in sigma]
    return np.array(gaussian_psfs) * oversample**2



class PSF:
    def __init__(self, n_subarr=1, n_subpix=13, oversample=10):
        self.n_subarr = n_subarr
        self.n_subpix = n_subpix
        self.oversample = oversample
        self.size = _round_up_to_odd_integer(oversample * (n_subpix + 1))
        self._psf = self._init_arr()
        
        # denote some information about the PSF maybe
        self.info = None
    
    @property
    def shape(self):
        return (self.size, self.size)
    
    def _init_arr(self):
        return np.zeros(self.shape)
    
    def normalize(self):
        pass
        
    def copy(self):
        return copy.deepcopy(self)
    
    def __add__(self, psf):
        new = self.copy()
        new += psf
        return new

    def __iadd__(self, psf):
        self._psf += psf._psf
        return self


    def __mul__(self, x):
        new = self.copy()
        new *= x
        return new

    def __imul__(self, x):
        if np.isscalar(x):
            self._psf *= x
        else:
            raise ValueError('Need to multiply by a scalar')
        return self
    
    # def __array__(self):
    # cant use this will return an array object rather than psf
    #     return self._psf
 
    __rmul__ = __mul__ # either way works
         
    # do other stuff
    
    def map_psf(self, coords):
        """makes the psflet that is on the detector"""        
        return map_coordinates(self._psf, coords, prefilter=False)

    
    def interp2d(self ):
        """Bilinear interpolation"""
        pass
    
    
class GaussianPSF(PSF):
    def __init__(self, wavelength, lam_ref, fwhm, n_subarr=1, n_subpix=13, oversample=10):
        super(GaussianPSF, self).__init__(n_subarr=n_subarr, 
                                          n_subpix=n_subpix, 
                                          oversample=oversample)
        sigma = fwhm * gaussian_fwhm_to_sigma * oversample * wavelength / lam_ref
        self._psf = Gaussian2DKernel(sigma, x_size=self.size).array * oversample**2
        
        
class PSFCube:
    def __init__(self, wavelengths):
        self._wavelengths = wavelengths
        self._psf_cube = None
        

    
    def interp(self, wavelength):
        """interpolate a psf model given a wavelength"""
        if hasattr(wavelength, "__len__"):
            return [self._interp(wlen) for wlen in wavelength]
        else:
            return self._interp(wavelength)
    
    def _interp(self, wavelength):
        """linear interpolation"""
        if wavelength <= np.min(self._wavelengths):
            return self[0]
        if wavelength >= np.max(self._wavelengths):
            return self[-1]
        else:
            # print(self._wavelengths, wavelength)
            lower_bound, upper_bound = nearest(self._wavelengths, wavelength)
            lam_0, lam_1 = self._wavelengths[lower_bound], self._wavelengths[upper_bound]
            psf_0, psf_1 = self[lower_bound], self[upper_bound]
            dlam_bound = lam_1 - lam_0
            weight = (wavelength - lam_0) / dlam_bound
            psf_new =  (1 - weight) * psf_0 + weight * psf_1
            return psf_new
    
    def __getitem__(self, idx):
        return self._psf_cube[idx]
    
    def copy(self):
        return copy.deepcopy(self)
        
class GaussianPSFCube(PSFCube):
    def __init__(self, wavelengths, lam_ref, fwhm, **kwargs):
        super(GaussianPSFCube, self).__init__(wavelengths=wavelengths)
        
        self._psf_cube = self._init_psfs(wavelengths, fwhm, lam_ref, **kwargs)
        
        # info
        self.description = "List of oversampled Gaussian PSFs"
        self.type = "GaussianCube"
    
    def _init_psfs(self, wavelength, fwhm, lam_ref, **kwargs):
        return [GaussianPSF(wlen, lam_ref, fwhm, **kwargs) for wlen in wavelength]

    
# to do https://github.com/astronomyk/SimCADO/blob/master/simcado/psf.py
class UserPSFCube(PSFCube):
    def __init__(self):
        pass

class UserPSF(PSF):
    def __init__(self, filename, fits_ext=None):
        # Will need to customize this class for our use for crispy
        if fits_ext is None:
            fits_ext = 0
        
        self.filename = filename
        
        header = fits.getheader(filename)
        data = fits.getdata(filename, ext=fits_ext)
        wavelengths = fits.getdata(filename, fits)
        
        
        self.header = header
        self._psf = data
        
        pass
