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
from .sefilter2 import *


def lpdec(x, h, g):
    """ LPDEC   Laplacian Pyramid Decomposition

    [c, d] = lpdec(x, h, g)

    Input:
    x:      input image
    h, g:   two lowpass filters for the Laplacian pyramid

    Output:
    c:      coarse image at half size
    d:      detail image at full size

    See also:	LPREC, PDFBDEC"""

    # Lowpass filter and downsample
    xlo = sefilter2(x, h, h, 'per', None)
    c = xlo[::2, ::2]

    # Compute the residual (bandpass) image by upsample, filter, and subtract
    # Even size filter needs to be adjusted to obtain perfect reconstruction
    adjust = mod(len(g) + 1, 2)

    xlo = zeros(x.shape)
    xlo[::2, ::2] = c
    d = x - sefilter2(xlo, g, g, 'per', adjust * array([1, 1]))

    return c, d
