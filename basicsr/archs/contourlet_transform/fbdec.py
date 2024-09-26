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
from .efilter2 import *
from .qdown import *
from .pdown import *
from .dfilters import *


def fbdec(x, h0, h1, type1, type2, extmod):
    """ FBDEC   Two-channel 2D Filterbank Decomposition

    [y0, y1] = fbdec(x, h0, h1, type1, type2, [extmod])

    Input:
    x:	input image
    h0, h1:	two decomposition 2D filters
    type1:	'q', 'p' or 'pq' for selecting quincunx or parallelogram
    downsampling matrix
    type2:	second parameter for selecting the filterbank type
    If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
    If type1 == 'p' then type2 is one of {0, 1, 2, 3}
    Those are specified in QDOWN and PDOWN
    If type1 == 'pq' then same as 'p' except that
    the paralellogram matrix is replaced by a combination
    of a  resampling and a quincunx matrices
    extmod:	[optional] extension mode (default is 'per')

    Output:
    y0, y1:	two result subband images

    Note:		This is the general implementation of 2D two-channel
    filterbank

    See also:	FBDEC_SP """

    if extmod is None:
        extmod = 'per'

    # For parallegoram filterbank using quincunx downsampling, resampling is
    # applied before filtering
    if type1 == 'pq':
        x = resamp(x, type2, None, None)

    # Stagger sampling if filter is odd-size (in both dimensions)
    if all(mod(h1.shape, 2)):
        shift = array([[-1], [0]])

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
    # Extend, filter and keep the original size
    y0 = efilter2(x, h0, extmod, None)
    y1 = efilter2(x, h1, extmod, shift)
    # Downsampling
    if type1 == 'q':
        # Quincunx downsampling
        y0 = qdown(y0, type2, None, None)
        y1 = qdown(y1, type2, None, None)
    elif type1 == 'p':
        # Parallelogram downsampling
        y0 = pdown(y0, type2, None, None)
        y1 = pdown(y1, type2, None, None)
    elif type1 == 'pq':
        # Quincux downsampling using the equipvalent type
        pqtype = ['1r', '2r', '2c', '1c']
        y0 = qdown(y0, pqtype[type2], None, None)
        y1 = qdown(y1, pqtype[type2], None, None)
    else:
        print("Invalid input type1")

    return y0, y1

#x = arange(1,26).reshape(5,5)
#fname = 'haar'
#h0, h1 = dfilters(fname, 'd')
#k0 = modulate2(h0, 'c', None)
#k1 = modulate2(h1, 'c', None)
#a,b = fbdec(x, k0, k1, 'q', '1c', 'qper_row')
# print a,b
