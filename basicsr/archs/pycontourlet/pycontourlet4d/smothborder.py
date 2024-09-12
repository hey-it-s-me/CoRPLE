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


def smothborder(x, n):
    """
    SMTHBORDER  Smooth the borders of a signal or image
    y = smothborder(x, n)

    Input:
    x:      the input signal or image
    n:      number of samples near the border that will be smoothed

    Output:
    y:      output image

    Note: This function provides a simple way to avoid border effect."""

    # Hamming window of size 2N
    w = 0.54 - 0.46 * cos(2 * pi * arange(0, 2 * n) / (2 * n - 1))

    if x.ndim == 1:
        W = ones(x.size)
        W[0:n] = w[0:n]
        W[-1 - n + 1::] = w[-1 - n + 1::]
        y = W.reshape(x.shape) * x
    elif x.ndim == 2:
        n1, n2 = x.shape
        W1 = ones((n1, 1))
        W1[0:n] = w[:, newaxis][0:n]
        W1[-1 - n + 1::] = w[:, newaxis][n::]

        y = W1 * ones((n2, n2)) * x

        W2 = ones(n2)
        W2[0:n] = w[0:n]
        W2[-1 - n + 1::] = w[n::]
        y = W2 * ones((n1, n1)) * y
    else:
        print 'First input must be a signal or image'

    return y
