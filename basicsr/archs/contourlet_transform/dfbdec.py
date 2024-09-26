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
from .dfilters import *
from .modulate2 import *
from .ffilters import *
from .backsamp import *
from .fbdec import *
from .dfbrec import *
import pdb


def dfbdec(x, fname, n):
    """ DFBDEC   Directional Filterbank Decomposition

    y = dfbdec(x, fname, n)

    Input:
    x:      input image
    fname:  filter name to be called by DFILTERS
    n:      number of decomposition tree levels

    Output:
    y:	    subband images in a cell vector of length 2^n

    Note:
    This is the general version that works with any FIR filters

    See also: DFBREC, FBDEC, DFILTERS"""

    if (n != round(n)) or (n < 0):
        print('Number of decomposition levels must be a non-negative integer')

    if n == 0:
        # No decomposition, simply copy input to output
        y = [None]
        y[0] = x.copy()
        return y

    # Get the diamond-shaped filters
    h0, h1 = dfilters(fname, 'd')

    # Fan filters for the first two levels
    # k0: filters the first dimension (row)
    # k1: filters the second dimension (column)
    k0 = modulate2(h0, 'c', None)
    k1 = modulate2(h1, 'c', None)
    # Tree-structured filter banks
    if n == 1:
        # Simplest case, one level
        y = [[None]] * 2
        y[0], y[1] = fbdec(x, k0, k1, 'q', '1r', 'per')
    else:
        # For the cases that n >= 2
        # First level
        x0, x1 = fbdec(x, k0, k1, 'q', '1r', 'per')
        # Second level
        y = [[None]] * 4
        y[0], y[1] = fbdec(x0, k0, k1, 'q', '2c', 'qper_col')
        y[2], y[3] = fbdec(x1, k0, k1, 'q', '2c', 'qper_col')
        # Fan filters from diamond filters
        f0, f1 = ffilters(h0, h1)
        # Now expand the rest of the tree
        for l in range(3, n + 1):
            # Allocate space for the new subband outputs
            y_old = y[:]
            y = [[None]] * 2**l
            # The first half channels use R1 and R2
            for k in range(0, 2**(l - 2)):
                i = mod(k, 2)
                y[2 * k], y[2 * k + 1] = fbdec(y_old[k],
                                               f0[i], f1[i], 'pq', i, 'per')
            # The second half channels use R3 and R4
            for k in range(2**(l - 2), 2**(l - 1)):
                i = mod(k, 2) + 2
                y[2 * k], y[2 * k + 1] = fbdec(y_old[k],
                                               f0[i], f1[i], 'pq', i, 'per')
    # Back sampling (so that the overal sampling is separable)
    # to enhance visualization
    y = backsamp(y)
    # Flip the order of the second half channels
    y[2**(n - 1)::] = y[::-1][:2**(n - 1)]

    return y