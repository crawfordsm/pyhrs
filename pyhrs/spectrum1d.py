# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the Spectrum1D class.

from __future__ import print_function, division

__all__ = ['Spectrum1D']

import copy
from astropy.extern import six
from astropy import log
from astropy.nddata import NDDataArray, FlagCollection

from astropy import units as u

import numpy as np


class Spectrum1D(NDDataArray):

    """A subclass of `NDData` for a one dimensional spectrum in Astropy.

    This class inherits all the base class functionality from the NDData class
    and has similar properties with the specutils.Spectrum1D object.

    This is a placeholder until specutils is released

    Parameters
    ----------
    wavelength : `~numpy.ndarray`
        wavelength of the spectrum

    data : `~numpy.ndarray`
        flux of the spectrum

    wcs : `~astropy.wcs.WCS`
        transformation between pixel coordinates and "dispersion" coordinates
        this carries the unit of the dispersion

    unit : `~astropy.unit.Unit` or None, optional
        unit of the flux, default=None

    mask : `~numpy.ndarray`, optional
        Mask for the data, given as a boolean Numpy array with a shape
        matching that of the data. The values must be ``False`` where
        the data is *valid* and ``True`` when it is not (like Numpy
        masked arrays). If `data` is a numpy masked array, providing
        `mask` here will causes the mask from the masked array to be
        ignored.

    meta : `dict`-like object, optional
        Metadata for this object.  "Metadata" here means all information that
        is included with this object but not part of any other attribute
        of this particular object.  e.g., creation date, unique identifier,
        simulation parameters, exposure time, telescope name, etc.

    """

    @classmethod
    def from_array(cls, dispersion, flux, dispersion_unit=None,
                   uncertainty=None, mask=None,
                   meta=None, copy=True, unit=None):
        return cls(wavelength=dispersion, flux=flux, wcs=None, unit=unit,
                   uncertainty=uncertainty, mask=mask, meta=meta)

    def __init__(self, wavelength, flux, wcs=None, unit=None, uncertainty=None,
                 mask=None, meta=None, indexer=None):

        super(Spectrum1D, self).__init__(data=flux, unit=unit, wcs=wcs,
                                         uncertainty=uncertainty,
                                         mask=mask, meta=meta)

        self.wavelength = wavelength

    @property
    def flux(self):
        return u.Quantity(self.data, self.unit, copy=False)

    @flux.setter
    def flux_setter(self, flux):
        if hasattr(flux, 'unit'):
            if self.unit is not None:
                flux = flux.to(self.unit).value
            else:
                raise ValueError('Attempting to set a new unit for this object'
                                 'this is not allowed by Spectrum1D')
        self.data = flux

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        if value is None:
            self._wavelength = None
        else:
            self._wavelength = value
