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
from ldfilter import *


def dfbdec_l(x, f, n):
    """ DFBDEC_L   Directional Filterbank Decomposition using Ladder Structure

    y = dfbdec_l(x, f, n)

    Input:
    x:	input image
    f:	filter in the ladder network structure,
    can be a string naming a standard filter (see LDFILTER)
    n:	number of decomposition tree levels

    Output:
    y:	subband images in a cell array (of size 2^n x 1)"""

    if (n != round(n)) or (n < 0):
        print 'Number of decomposition levels must be a non-negative integer'

    if n == 0:
        # No decomposition, simply copy input to output
        y[0] = x
        return y

    # Ladder filter
    if str(f) == f:
        f = ldfilter(f)

    # Tree-structured filter banks
    if n == 1:
        # Simplest case, one level
        y[0], y[1] = fbdec_l(x, f, 'q', '1r', 'qper_col')
    else:
        # For the cases that n >= 2
        # First level
        x0, x1 = fbdec_l(x, f, 'q', '1r', 'qper_col')

        # Second level
        y = [[None]] * 4
        y[1], y[0] = fbdec_l(x0, f, 'q', '2c', 'per')
        y[3], y[2] = fbdec_l(x1, f, 'q', '2c', 'per')

        # Now expand the rest of the tree
        for l in xrange(2, n):
            # Allocate space for the new subband outputs
            y_old = y[:]
            y = [[None]] * 2**l

            # The first half channels use R1 and R2
            for k in xrange(0, 2**(l - 2)):
                i = mod(k - 1, 2) + 1
                y[2 * k], y[2 * k - 1] = fbdec_l(y_old[k], f, 'p', i, 'per')

                # The second half channels use R3 and R4
            for k in xrange(2**(l - 2) + 1, 2**(l - 1)):
                i = mod(k - 1, 2) + 3
                y[2 * k], y[2 * k - 1] = fbdec_l(y_old[k], f, 'p', i, 'per')

    # Backsampling
    y = backsamp(y)

    # Flip the order of the second half channels
    y[2**(n - 1) + 1::] = fliplr(y[2**(n - 1) + 1::])
    return y
