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
from .resampz import *


def qupz(x, type):
    """ QUPZ   Quincunx Upsampling (with zero-pad and matrix extending)
        y = qup(x, [type])
        Input:
        x:	input image
        type:	[optional] 1 or 2 for selecting the quincunx matrices:
                        Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
        Output:
        y:	qunincunx upsampled image

       This resampling operation does NOT involve periodicity, thus it
       zero-pad and extend the matrix"""

    if type is None:
        type = 0
    """ Quincunx downsampling using the Smith decomposition:
	Q1 = R2 * [2, 0; 0, 1] * R3
        and,
	Q2 = R1 * [2, 0; 0, 1] * R4

        See RESAMP for the definition of those resampling matrices

        Note that R1 * R2 = R3 * R4 = I so for example,
        upsample by R1 is the same with down sample by R2.
        Also the order of upsampling operations is in the reserved order
        with the one of matrix multiplication."""

    if type == 0:
        x1 = resampz(x, 3, None)
        m, n = x1.shape
        x2 = zeros([2 * m - 1, n])
        x2[::2, :] = x1.copy()
        y = resampz(x2, 0, None)
        return y

    elif type == 1:
        x1 = resampz(x, 2, None)
        m, n = x1.shape
        x2 = zeros([2 * m - 1, n])
        x2[::2, :] = x1.copy()
        y = resampz(x2, 1, None)
        return y
