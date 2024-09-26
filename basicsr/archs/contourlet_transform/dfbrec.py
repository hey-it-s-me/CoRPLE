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
from .ffilters import *
from .modulate2 import *
from .fbrec import *
from .rebacksamp import *


def dfbrec(y, fname):
    """ DFBREC   Directional Filterbank Reconstruction

    x = dfbrec(y, fname)

    Input:
    y:	    subband images in a cell vector of length 2^n
    fname:  filter name to be called by DFILTERS

    Output:
    x:	    reconstructed image

    See also: DFBDEC, FBREC, DFILTERS"""

    n = int(log2(len(y)))

    if (n != round(n)) or (n < 0):
        print("Number of reconstruction levels must be a non-negative integer")

    if n == 0:
        # Simply copy input to output
        x = [None]
        x[0] = y[0][:]
        return x

    # Get the diamond-shaped filters
    h0, h1 = dfilters(fname, 'r')

    # Fan filters for the first two levels
    # k0: filters the first dimension (row)
    # k1: filters the second dimension (column)
    k0 = modulate2(h0, 'c', None)
    k1 = modulate2(h1, 'c', None)

    # Flip back the order of the second half channels
    y[2**(n - 1)::] = y[::-1][:2**(n - 1)]

    # Undo backsampling
    y = rebacksamp(y)

    # Tree-structured filter banks

    if n == 1:
        # Simplest case, one level
        x = fbrec(y[0], y[1], k0, k1, 'q', '1r', 'per')
    else:
        # For the cases that n >= 2
        # Fan filters from diamond filters
        f0, f1 = ffilters(h0, h1)

        # Recombine subband outputs to the next level

        for l in xrange(n, 2, -1):
            y_old = y[:]
            y = [[None]] * 2**(l - 1)

            # The first half channels use R1 and R2
            for k in xrange(0, 2**(l - 2)):
                i = mod(k, 2)
                y[k] = fbrec(y_old[2 * k], y_old[2 * k + 1],
                             f0[i], f1[i], 'pq', i, 'per')
            # The second half channels use R3 and R4
            for k in xrange(2**(l - 2), 2**(l - 1)):
                i = mod(k - 1, 2) + 2
                y[k] = fbrec(y_old[2 * k], y_old[2 * k + 1],
                             f0[i], f1[i], 'pq', i, 'per')

        # Second level
        x0 = fbrec(y[0], y[1], k0, k1, 'q', '2c', 'qper_col')
        x1 = fbrec(y[2], y[3], k0, k1, 'q', '2c', 'qper_col')

        # First level
        x = fbrec(x0, x1, k0, k1, 'q', '1r', 'per')

    return x
