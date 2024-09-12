# -*- coding: utf-8 -*-
#    PyContourlet
#
#    A Python library for the Contourlet Transform.
#
#    Copyright (C) 2011 Mazay Jiménez
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation version 2.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from numpy import *
from .qup import *
from .efilter2 import *
from .resamp import *


def fbrec(y0, y1, h0, h1, type1, type2, extmod):
    """ FBREC   Two-channel 2D Filterbank Reconstruction

    x = fbrec(y0, y1, h0, h1, type1, type2, [extmod])

    Input:
    y0, y1:	two input subband images
    h0, h1:	two reconstruction 2D filters
    type1:	'q', 'p' or 'pq' for selecting quincunx or parallelogram
    upsampling matrix
    type2:	second parameter for selecting the filterbank type
    If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
    If type1 == 'p' then type2 is one of {0, 1, 2, 3}
    Those are specified in QUP and PUP
    If type1 == 'pq' then same as 'p' except that
    the paralellogram matrix is replaced by a combination
    of a quincunx and a resampling matrices
    extmod:	[optional] extension mode (default is 'per')

    Output:
    x:	reconstructed image

    Note:	This is the general case of 2D two-channel filterbank

    See also:	FBDEC"""

    if extmod is None:
        extmod = 'per'

    # Upsampling
    if type1 == 'q':
        # Quincunx upsampling
        y0 = qup(y0, type2, None)
        y1 = qup(y1, type2, None)
    elif type1 == 'p':
        # Parallelogram upsampling
        y0 = pup(y0, type2, None)
        y1 = pup(y1, type2, None)
    elif type1 == 'pq':
        # Quincux upsampling using the equivalent type
        pqtype = ['1r', '2r', '2c', '1c']
        y0 = qup(y0, pqtype[type2], None)
        y1 = qup(y1, pqtype[type2], None)
    else:
        print("Invalid input type1")

    # Stagger sampling if filter is odd-size
    if all(mod(h1.shape, 2)):
        shift = array([[1], [0]])

    # Account for the resampling matrix in the parallegoram case
    if type1 == 'p':
        R = [[None]] * 4
        R[0] = array([[1, 1], [0, 1]])
        R[1] = array([[1, -1], [0, 1]])
        R[2] = array([[1, 0], [1, 1]])
        R[3] = array([[1, 0], [-1, 1]])
        shift = R[type2] * shift
    else:
        shift = array([[0], [0]])

    # Dimension that has even size filter needs to be adjusted to obtain
    # perfect reconstruction with zero shift
    adjust0 = mod(array([h0.shape]) + 1, 2).conj().T
    adjust1 = mod(array([h1.shape]) + 1, 2).conj().T

    # Extend, filter and keep the original size
    x0 = efilter2(y0, h0, extmod, adjust0)
    x1 = efilter2(y1, h1, extmod, adjust1 + shift)

    # Combine 2 channel to output
    x = x0 + x1

    # For parallegoram filterbank using quincunx upsampling,
    # a resampling is required at the end
    if type1 == 'pq':
        # Inverse of resamp(x, type)
        inv_type = [1, 0, 3, 2]
        x = resamp(x, inv_type[type2], None, None)

    return x
