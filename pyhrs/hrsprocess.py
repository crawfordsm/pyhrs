# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base processing for HRS data
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
from astropy.extern import six

from astropy import units as u

import ccdproc

__all__ = ['ccd_process', 'create_masterbias', 'hrs_process', 'blue_process',
           'red_process']


def ccd_process(ccd, oscan=None, trim=None, error=False, masterbias=None,
                bad_pixel_mask=None, gain=None, rdnoise=None,
                oscan_median=True, oscan_model=None):
    """Perform basic processing on ccd data.

       The following steps can be included:
        * overscan correction
        * trimming of the image
        * create edeviation frame
        * gain correction
        * add a mask to the data
        * subtraction of master bias

       The task returns a processed `ccdproc.CCDData` object.

    Parameters
    ----------
    ccd: `ccdproc.CCDData`
        Frame to be reduced

    oscan: None, str, or, `~ccdproc.ccddata.CCDData`
        For no overscan correction, set to None.   Otherwise proivde a region
        of `ccd` from which the overscan is extracted, using the FITS
        conventions for index order and index start, or a
        slice from `ccd` that contains the overscan.

    trim: None or str
        For no trim correction, set to None.   Otherwise proivde a region
        of `ccd` from which the image should be trimmed, using the FITS
        conventions for index order and index start.

    error: boolean
        If True, create an uncertainty array for ccd

    masterbias: None, `~numpy.ndarray`,  or `~ccdproc.CCDData`
        A materbias frame to be subtracted from ccd.

    bad_pixel_mask: None or `~numpy.ndarray`
        A bad pixel mask for the data. The bad pixel mask should be in given
        such that bad pixels havea value of 1 and good pixels a value of 0.

    gain: None or `~astropy.Quantity`
        Gain value to multiple the image by to convert to electrons

    rdnoise: None or `~astropy.Quantity`
        Read noise for the observations.  The read noise should be in
        `~astropy.units.electron`


    oscan_median :  bool, optional
        If true, takes the median of each line.  Otherwise, uses the mean

    oscan_model :  `~astropy.modeling.Model`, optional
        Model to fit to the data.  If None, returns the values calculated
        by the median or the mean.

    Returns
    -------
    ccd: `ccdproc.CCDData`
        Reduded ccd

    Examples
    --------

    1. To overscan, trim, and gain correct a data set:

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from hrsprocess import ccd_process
    >>> ccd = CCDData(np.ones([100, 100]), unit=u.adu)
    >>> nccd = ccd_process(ccd, oscan='[1:10,1:100]', trim='[10:100, 1,100]',
                           error=False, gain=2.0*u.electron/u.adu)


    """
    # make a copy of the object
    nccd = ccd.copy()

    # apply the overscan correction
    if isinstance(oscan, ccdproc.CCDData):
        nccd = ccdproc.subtract_overscan(nccd, overscan=oscan,
                                         median=oscan_median,
                                         model=oscan_model)
    elif isinstance(oscan, six.string_types):
        nccd = ccdproc.subtract_overscan(nccd, fits_section=oscan,
                                         median=oscan_median,
                                         model=oscan_model)
    elif oscan is None:
        pass
    else:
        raise TypeError('oscan is not None, a string, or CCDData object')

    # apply the trim correction
    if isinstance(trim, six.string_types):
        nccd = ccdproc.trim_image(nccd, fits_section=trim)
    elif trim is None:
        pass
    else:
        raise TypeError('trim is not None or a string')

    # create the error frame
    if error and gain is not None and rdnoise is not None:
        nccd = ccdproc.create_deviation(nccd, gain=gain, rdnoise=rdnoise)
    elif error and (gain is None or rdnoise is None):
        raise ValueError(
            'gain and rdnoise must be specified to create error frame')

    # apply the bad pixel mask
    if isinstance(bad_pixel_mask, np.ndarray):
        nccd.mask = bad_pixel_mask
    elif bad_pixel_mask is None:
        pass
    else:
        raise TypeError('bad_pixel_mask is not None or numpy.ndarray')

    # apply the gain correction
    if isinstance(gain, u.quantity.Quantity):
        nccd = ccdproc.gain_correct(nccd, gain)
    elif gain is None:
        pass
    else:
        raise TypeError('gain is not None or astropy.Quantity')

    # test subtracting the master bias
    if isinstance(masterbias, ccdproc.CCDData):
        nccd = nccd.subtract(masterbias)
    elif isinstance(masterbias, np.ndarray):
        nccd.data = nccd.data - masterbias
    elif masterbias is None:
        pass
    else:
        raise TypeError(
            'masterbias is not None, numpy.ndarray,  or a CCDData object')

    return nccd


def hrs_process(image_name, ampsec=[], oscansec=[], trimsec=[],
                masterbias=None, error=False, bad_pixel_mask=None, flip=False,
                rdnoise=None, oscan_median=True, oscan_model=None):
    """Processing required for HRS observations.  If the images have multiple
       amps, then this will process each part of the image and recombine them
       into for the final results

    Parameters
    ----------
    image_name: str
       Name of file to be processed

    ampsec: list
       List of ampsections.  This list should have the same length as the
       number of amps in the data set.  The sections should be given
       in the format of fits_sections (see below).

    oscansec: list
       List of overscan sections.  This list should have the same length as the
       number of amps in the data set.  The sections should be given
       in the format of fits_sections (see below).

    trimsec: list
       List of overscan sections.  This list should have the same length as the
       number of amps in the data set.  The sections should be given
       in the format of fits_sections (see below).

    error: boolean
        If True, create an uncertainty array for ccd

    masterbias: None, `~numpy.ndarray`,  or `~ccdproc.CCDData`
        A materbias frame to be subtracted from ccd.

    bad_pixel_mask: None or `~numpy.ndarray`
        A bad pixel mask for the data. The bad pixel mask should be in given
        such that bad pixels havea value of 1 and good pixels a value of 0.


    flip: boolean
        If True, the image will be flipped such that the orders run from the
        bottom of the image to the top and the dispersion runs from the left
        to the right.

    rdnoise: None or `~astropy.Quantity`
        Read noise for the observations.  The read noise should be in
        `~astropy.units.electron`

    oscan_median :  bool, optional
        If true, takes the median of each line.  Otherwise, uses the mean

    oscan_model :  `~astropy.modeling.Model`, optional
        Model to fit to the data.  If None, returns the values calculated
        by the median or the mean.

    Returns
     -------
    ccd: `~ccdproc.CCDData`
        Data processed and


    Notes
    -----

    The format of the `fits_section` string follow the rules for slices that
    are consistent with the FITS standard (v3) and IRAF usage of keywords like
    TRIMSEC and BIASSEC. Its indexes are one-based, instead of the
    python-standard zero-based, and the first index is the one that increases
    most rapidly as you move through the array in memory order, opposite the
    python ordering.

    The 'fits_section' argument is provided as a convenience for those who are
    processing files that contain TRIMSEC and BIASSEC. The preferred, more
    pythonic, way of specifying the overscan is to do it by indexing the data
    array directly with the `overscan` argument.

    """
    # read in the data
    ccd = ccdproc.CCDData.read(image_name, unit=u.adu)

    namps = ccd.header['CCDAMPS']
    # thow errors for the wrong number of amps
    if len(ampsec) != namps:
        raise ValueError('Number of ampsec does not equal number of amps')
    if len(oscansec) != namps:
        raise ValueError('Number of oscansec does not equal number of amps')
    if len(trimsec) != namps:
        raise ValueError('Number of trimsec does not equal number of amps')

    if namps == 1:
        gain = float(ccd.header['gain'].split()[0]) * u.electron / u.adu
        nccd = ccd_process(ccd, oscan=oscansec[0], trim=trimsec[0],
                           error=error, masterbias=masterbias,
                           bad_pixel_mask=bad_pixel_mask, gain=gain,
                           rdnoise=rdnoise, oscan_median=oscan_median,
                           oscan_model=oscan_model)
    else:
        ccd_list = []
        xsize = 0
        for i in range(namps):
            cc = ccdproc.trim_image(ccd, fits_section=ampsec[i])

            gain = float(ccd.header['gain'].split()[i]) * u.electron / u.adu
            ncc = ccd_process(cc, oscan=oscansec[i], trim=trimsec[i],
                              error=False, masterbias=None, gain=gain,
                              bad_pixel_mask=None, rdnoise=rdnoise,
                              oscan_median=oscan_median,
                              oscan_model=oscan_model)
            xsize = xsize + ncc.shape[1]
            ysize = ncc.shape[0]
            ccd_list.append(ncc)

        # now recombine the processed data
        ncc = ccd_list[0]
        data = np.zeros((ysize, xsize))
        if ncc.mask is not None:
            mask = np.zeros((ysize, xsize))
        else:
            mask = None
        if ncc.uncertainty is not None:
            raise NotImplementedError(
                'Support for uncertainties not implimented yet')
        else:
            uncertainty = None

        x1 = 0
        for i in range(namps):
            x2 = x1 + ccd_list[i].data.shape[1]
            data[:, x1:x2] = ccd_list[i].data
            if mask is not None:
                mask[:, x1:x2] = ccd_list[i].mask
            x1 = x2

        nccd = ccdproc.CCDData(data, unit=ncc.unit, mask=mask,
                               uncertainty=uncertainty)
        nccd = ccd_process(nccd, masterbias=masterbias, error=error, gain=None,
                           rdnoise=rdnoise, bad_pixel_mask=bad_pixel_mask)

    if flip:
        nccd.data = nccd.data[::-1, ::-1]
        if (nccd.mask is not None):
            nccd.mask = nccd.mask[::-1, ::-1]
        if (nccd.uncertainty is not None):
            raise NotImplementedError(
                'Flipping is not implimented yet for uncertainty')

    return nccd


def blue_process(infile, masterbias=None, error=False, rdnoise=None):
    """Process a blue frame
    """
    # check to make sure it is a blue file

    # reduce file
    blueamp = ['[1:1050,:]', '[1051:2100,:]']
    bluescan = ['[1:26,:]', '[1025:1050,:]']
    bluetrim = ['[27:1050,:]', '[1:1024,:]']
    ccd = hrs_process(infile, ampsec=blueamp, oscansec=bluescan,
                      trimsec=bluetrim, masterbias=masterbias, error=error,
                      rdnoise=None, flip=True)
    return ccd


def red_process(infile, masterbias=None, error=None, rdnoise=None):
    """Process a blue frame
    """
    redamp = ['[1:4122,1:4112]']
    redscan = ['[1:25,1:4112]']
    redtrim = ['[27:4122,1:4112]']
    ccd = hrs_process(infile, ampsec=redamp, oscansec=redscan,
                      trimsec=redtrim, masterbias=masterbias, error=error,
                      rdnoise=None, flip=False)
    return ccd


def create_masterbias(image_list):
    """Create a master bias frame from a list of images

    Parameters
    ----------
    image_list: list
        List contain the file names to be processed

    Returns
    -------
    masterbias: ccddata.CCDData
        Combine master bias from the biases supplied in image_list

    """
    # determine whether they are red or blue
    if os.path.basename(image_list[0]).startswith('H'):
        func = blue_process
    elif os.path.basename(image_list[0]).startswith('R'):
        func = red_process
    else:
        raise TypeError('These are not standard HRS frames')

    # reduce the files
    ccd_list = []
    for image_name in image_list:
        ccd = func(image_name, masterbias=None, error=False)
        ccd_list.append(ccd)

    # combine the files
    cb = ccdproc.Combiner(ccd_list)
    nccd = cb.median_combine(median_func=np.median)

    return nccd
