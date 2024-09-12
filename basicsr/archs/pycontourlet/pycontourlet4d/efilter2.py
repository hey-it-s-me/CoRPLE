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
from .extend2 import *
from scipy import signal


def efilter2(x, f, extmod, shift):
    """EFILTER2   2D Filtering with edge handling (via extension)

    y = efilter2(x, f, [extmod], [shift])

    Input:
    x:	input image
    f:	2D filter
    extmod:	[optional] extension mode (default is 'per')
    shift:	[optional] specify the window over which the
    convolution occurs. By default shift = [0; 0].

    Output:
    y:	filtered image that has:
    Y(z1,z2) = X(z1,z2)*F(z1,z2)*z1^shift(1)*z2^shift(2)

    Note:
    The origin of filter f is assumed to be floor(size(f)/2) + 1.
    Amount of shift should be no more than floor((size(f)-1)/2).
    The output image has the same size with the input image.

    See also:	EXTEND2, SEFILTER2"""

    if extmod is None:
        extmod = 'per'
    if shift is None:
        shift = array([[0], [0]])

    # Periodized extension
    if f.ndim < 2:
        sf = (r_[1, array(f.shape)] - 1) / 2.0
    else:
        sf = (array(f.shape) - 1) / 2.0

    ru = int(floor(sf[0]) + shift[0][0])
    rd = int(ceil(sf[0]) - shift[0][0])
    cl = int(floor(sf[1]) + shift[1][0])
    cr = int(ceil(sf[1]) - shift[1][0])
    xext = extend2(x, ru, rd, cl, cr, extmod)

    # Convolution and keep the central part that has the size as the input
    if f.ndim < 2:
        y = signal.convolve(xext.T, f[:, newaxis], 'valid').T
    else:
        y = signal.convolve(xext, f, 'valid')

    return y

#x = arange(1,26).reshape(5,5)
#f = array([-0.70710678,  0.70710678])
#extmod = 'qper_col'
#shift = array([[0], [0]])
#a = efilter2(x, f, extmod, shift)
# print a
