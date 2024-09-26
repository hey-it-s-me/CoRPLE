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
from scipy import signal
from .extend2 import *


def sefilter2(x, f1, f2, extmod, shift):
    """SEFILTER2   2D separable filtering with extension handling
    y = sefilter2(x, f1, f2, [extmod], [shift])

    Input:
    x:      input image
    f1, f2: 1-D filters in each dimension that make up a 2D seperable filter
    extmod: [optional] extension mode (default is 'per')
    shift:  [optional] specify the window over which the
    convolution occurs. By default shift = [0; 0].

    Output:
    y:      filtered image of the same size as the input image:
    Y(z1,z2) = X(z1,z2)*F1(z1)*F2(z2)*z1^shift(1)*z2^shift(2)

    Note:
    The origin of the filter f is assumed to be floor(size(f)/2) + 1.
    Amount of shift should be no more than floor((size(f)-1)/2).
   The output image has the same size with the input image.

   See also: EXTEND2, EFILTER2"""

    if extmod is None:
        extmod = 'per'

    if shift is None:
        shift = array([[0], [0]])

    # Make sure filter in a row vector
    f1 = f1[:, newaxis].reshape(len(f1),)
    f2 = f2[:, newaxis].reshape(len(f2),)

    # Periodized extension
    lf1 = (len(f1) - 1) / 2.0
    lf2 = (len(f1) - 1) / 2.0

    y = extend2(x, floor(lf1) + shift[0], ceil(lf1) - shift[0],
                floor(lf2) + shift[1], ceil(lf2) - shift[1], extmod)

    # Seperable filter
    y = signal.convolve(y, f1[:, newaxis] * f2, 'valid')

    return y
