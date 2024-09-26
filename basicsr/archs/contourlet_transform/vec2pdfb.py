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
from matplotlib.mlab import find


def vec2pdfb(c, s):
    """ VEC2PDFB   Convert the vector form to the output structure of the PDFB

       y = vec2pdfb(c, s)

       Input:
       c:  1-D vector that contains all PDFB coefficients
       s:  structure of PDFB output

       Output:
       y:  PDFB coefficients in cell vector format that can be used in pdfbrec

       See also:	PDFB2VEC, PDFBREC"""

    # Copy the coefficients from c to y according to the structure s
    n = s[-1, 1]      # number of pyramidal layers
    y = [[None]] * n

    # Variable that keep the current position
    pos = prod(s[0, 2::])
    y[0] = c[0:pos].reshape(s[0, 2::])
    # Used for row index of s
    ind = 1

    for l in xrange(1, n):
        # Number of directional subbands in this layer
        nd = len(find(s[:, 0] == l))
        y[l] = [[None]] * nd
        for d in xrange(0, nd):
            # Size of this subband
            p = s[ind + d, 2]
            q = s[ind + d, 3]
            ss = p * q
            y[l][d] = c[pos + arange(0, ss)].reshape([p, q])
            pos = pos + ss
        ind = ind + nd

    return y


# c = array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  3,
#  3,  3,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,
#  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,
#  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4])

# s = array([[0, 0, 3, 3],
#       [1, 0, 2, 2],
#       [1, 1, 2, 2],
#       [1, 2, 2, 2],
#       [2, 0, 4, 4],
#       [2, 1, 4, 4],
#       [2, 2, 4, 4],
#       [2, 3, 4, 4]])

#y = vec2pdfb(c,s)

# print y
