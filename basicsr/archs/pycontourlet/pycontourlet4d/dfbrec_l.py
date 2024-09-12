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
from rebacksamp import *
from ldfilter import *
from fbrec_l import *


def dfbrec_l(y, f):
    """ DFBREC_L   Directional Filterbank Reconstruction using Ladder Structure

    x = dfbrec_l(y, fname)

    Input:
    y:	subband images in a cell vector of length 2^n
    f:	filter in the ladder network structure,
    can be a string naming a standard filter (see LDFILTER)

    Output:
    x:	reconstructed image

    See also:	DFBDEC, FBREC, DFILTERS"""

    n = log2(len(y))

    if (n != round(n)) or (n < 0):
        print 'Number of reconstruction levels must be a non-negative integer'

    if n == 0:
        # Simply copy input to output
        x = y[0]
        return x

    # Ladder filter
    if str(f) == f:
        f = ldfilter(f)

    # Flip back the order of the second half channels
    y[2**(n - 1) + 1::] = fliplr(y[2**(n - 1) + 1::])

    # Undo backsampling
    y = rebacksamp(y)

    # Tree-structured filter banks
    if n == 1:
        # Simplest case, one level
        x = fbrec_l(y]0], y[1], f, 'q', '1r', 'qper_col')
    else:
        # For the cases that n >= 2
        # Recombine subband outputs to the next level
        for l in xrange(n, 2, -1):
            y_old = y[: ]
            y= [[None]] * 2**(l - 1)

            # The first half channels use R1 and R2
            for k in xrange(0, 2**(l - 2)):
                i= mod(k - 1, 2) + 1
                y[k]= fbrec_l(y_old[2 * k], y_old[2 * k - 1], f, 'p', i, 'per')

            # The second half channels use R3 and R4
            for k in xrange(2**(l - 2) + 1, 2**(l - 1)):
                i= mod(k - 1, 2) + 3
                y[k]= fbrec_l(y_old[2 * k], y_old[2 * k - 1], f, 'p', i, 'per')
    # Second level
    x0= fbrec_l(y[1], y[0], f, 'q', '2c', 'per')
    x1= fbrec_l(y[3], y[2], f, 'q', '2c', 'per')

    # First level
    x= fbrec_l(x0, x1, f, 'q', '1r', 'qper_col')

   return x
