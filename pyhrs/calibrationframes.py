# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base processing for HRS data
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
from astropy.extern import six
import warnings

from astropy import units as u
from astropy import modeling as mod

from scipy import ndimage as nd


import ccdproc

from .hrsprocess import *
from .hrstools import *

__all__ = ['create_masterbias', 'create_masterflat', 'create_orderframe',
           'normalize_image']


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


def create_masterflat(image_list, masterbias=None):
    """Create a master flat frame from a list of images

    Parameters
    ----------
    image_list: list
        List contain the file names to be processed

    masterbias: None, `~numpy.ndarray`,  or `~ccdproc.CCDData`
        A materbias frame to be subtracted from ccd.

    Returns
    -------
    masterflat: ccddata.CCDData
        Combine master flat from the flats supplied in image_list

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
        ccd = func(image_name, masterbias=masterbias, error=False)
        ccd_list.append(ccd)

    # combine the files
    cb = ccdproc.Combiner(ccd_list)
    nccd = cb.median_combine(median_func=np.median)

    return nccd


def normalize_image(data, func_init, mask,
                    fitter=mod.fitting.LinearLSQFitter,
                    normalize=True):
    """Normalize an HRS image.

    The tasks takes an image and will fit a function to the overall
    shape to it.  The task will only fit to the illuminated orders
    and if an order_frame is provided it will use that to identify the
    areas it should fit to.  Otherwise, it will filter the image such that
    only the maximum areas are fit.

    This function will then be divided out of the image and return
    a normalized image if requested.

    Parameters
    ----------
    data: numpy.ndarray
         Data to be normalized

    mask: numpy.ndarray
         If a numpy.ndarray, this will be used to determine areas
         to be used for the fit.

    func_init: ~astropy.modeling.models
         Function to fit to the image

    fitter: ~astropy.modeling.fitting
         Fitter function

    normalize: boolean
         If normalize is True, it will return data normalized by the
         function fit to it.  If normalize is False, it will return
         an array representing the function fit to data.

    Returns
    -------
    ndata: numpy.ndarray
         If normalize is True, it will return data normalized by the
         function fit to it.  If normalize is False, it will return
         an array representing the function fit to data.

    """

    if isinstance(mask, np.ndarray):
        if mask.shape != data.shape:
            raise ValueError('mask is not the same shape as data')
    else:
        raise TypeError('mask is not None or an numpy.ndarray')

    ys, xs = data.shape
    if isinstance(fitter, mod.fitting._FitterMeta):
        g_fit = fitter()
    else:
        raise TypeError('fitter is not a valid astropy.modeling.fitting')

    if not hasattr(func_init, 'n_inputs'):
        raise TypeError('func_init is not a valid astropy.modeling.model')

    if func_init.n_inputs == 2:
        y, x = np.indices((ys, xs))
        f = g_fit(func_init, x[mask], y[mask], data[mask])
        ndata = f(x, y)
    elif func_init.n_inputs == 1:
        ndata = 0.0 * data
        xarr = np.arange(xs)
        yarr = np.arange(ys)
        for i in range(xs):
            f = g_fit(func_init, yarr[mask[:, i]], data[:, i][mask[:, i]])
            ndata[:, i] = f(yarr)

    if normalize:
        return data / ndata * ndata.mean()
    else:
        return ndata


def create_orderframe(data, first_order, xc, detect_kernal, smooth_length=15,
                      y_start=0, y_limit=None):
    """Create an order frame from from an observation.

    A two dimensional detect_kernal is correlated with the image.  The
    kernal steps through y-space until a match is made.  Once a best fit is
    found, the order is extracted to include pixels that may not be part of
    the initial detection kernal.  Once all pixels have been extracted, they
    are set to zero in the original frame.  The detection kernal is updated
    by the new order detected

    Parameters
    ----------
    data: ~numpy.ndarray
        An image with the different orders illuminated.  Any processing of this
        image should have been performed prior to running `create_orderframe`.

    first_order: int
        The first order to appear in the image starting from the bottom of the
        image

    xc: int
        The x-position to extract a 1-D map of the orders

    detect_kern: ~numpy.ndarray
        The initial detection kernal which have the shape of a single order.

    smooth_length: int
        The length to smooth the images by prior to processing them

    y_start: int
        The initial value to start searching for the first maximum

    y_limit: int
        The limit in y-positions for automatically finding the orders.


    Returns
    -------
    order_frame: ~numpy.ndarray
        An image with each of the order identified by their number

    Notes
    -----
    Currently no orders are extrcted above y_limit and the code still needs to
    be updated to handle those higher orders

    """
    # set up the arrays needed
    sdata = 1.0 * data
    ndata = data[:, xc]
    order_frame = 0.0 * data

    # set up additiona information that we need
    ys, xs = data.shape
    xc = int(0.5 * xs)
    norder = first_order

    if y_limit is None:
        y_limit = ys

    # convolve with the default kernal
    cdata = np.convolve(ndata, detect_kernal, mode='same')
    cdata *= (ndata > 0)
    cdata = nd.filters.maximum_filter(cdata, smooth_length)

    i = y_start
    nlen = len(detect_kernal)
    max_value = sdata.max()
    cdata[:y_start] = -1
    while i < ys:
        # find the highest peak in the convolution area
        y1 = max(0, i)
        y2 = y1 + nlen
        try:
            y2 = np.where(cdata == 0)[0][0]
        except Exception as e:
            warnings.warn(str(e))
        y2 = min(ys - 1, y2)
        try:
            yc = cdata[y1:y2].argmax() + y1
        except:
            warning.warn('Breakying at %i' % i)
            break
        # this is to make sure the two fibers
        # are both contained in the same
        # order
        sy1 = max(0, yc - 0.5 * smooth_length)
        sy2 = min(ys - 1, yc + 2 * smooth_length)
        sdata[sy1:sy2, xc] = max_value

        obj, nobj = nd.label(sdata > 0.5 * max_value)
        nobj = obj[yc, xc]
        order_frame += norder * (obj == nobj)

        # remove the data for this peak and
        # all data it is now safe to ignore
        cdata[0:y2] = -1
        # now remove everything up until the next
        # peak
        try:
            n2 = np.where(cdata > 0)[0][0]
            cdata[0:n2] = -1
        except:
            n2 = y2
        # change the smoothing length
        z = (order_frame == norder)[:, xc]
        smooth_length = 0.2 * z[z == 1].sum()

        i = n2
        norder += 1
        if i > y_limit:
            break
            # TODO: Something needs to be figured out for when above the limit
            sdata[order_frame > 0] = 0
            ndata = sdata[:, xc]
            detect_kernal = (order_frame == norder - 1)[:, xc]
            detect_kernal = detect_kernal[detect_kernal > 0]
            nlen = len(detect_kernal)

            if nlen > 0:
                cdata = np.convolve(ndata, detect_kernal, mode='same')
                cdata *= (ndata > 0)
                cdata = nd.filters.maximum_filter(cdata, smooth_length)
                yn = np.where(cdata > 0)[0][0]
                cdata[:yn] = -1
            y_limit = ys

    return order_frame
