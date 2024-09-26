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


def pdfb2vec(y):
    """ PDFB2VEC   Convert the output of the PDFB into a vector form

    [c, s] = pdfb2vec(y)

    Input:
    y:  an output of the PDFB

    Output:
    c:  1-D vector that contains all PDFB coefficients
    s:  structure of PDFB output, which is a four-column matrix.  Each row
    of s corresponds to one subband y{l}{d} from y, in which the first two
    entries are layer index l and direction index d and the last two
    entries record the size of y{l}{d}.

    See also:	PDFBDEC, VEC2PDFB"""

    n = len(y)

    # Save the structure of y into s
    temp = a[0].shape
    s = []
    s.append([0, 0, temp[0], temp[1]])

    # Used for row index of s
    ind = 0
    for l in xrange(1, n):
        nd = len(y[l])
        for d in xrange(0, nd):
            temp = y[l][d].shape
            s.extend([[l, d, temp[0], temp[1]]])
    ind = ind + nd

    s = array(s)
    # The total number of PDFB coefficients
    nc = sum(prod(s[:, 2::], axis=1))

    # Assign the coefficients to the vector c
    c = zeros(nc)

    # Variable that keep the current position
    pos = prod(y[0].shape)

    # Lowpass subband
    c[0:pos] = y[0].flatten(1)

    # Bandpass subbands
    for l in xrange(1, n):
        for d in xrange(0, len(y[l])):
            ss = prod(y[l][d].shape)
            c[pos + arange(0, ss)] = y[l][d]
            pos = pos + ss

    return c, s

#a  =  [[None]]*3
#a[0] = ones((3,3))
#a[1] = [[None]]*3
#a[1][0] = ones((2,2))
#a[1][1] = 2*ones((2,2))
#a[1][2] = 3*ones((2,2))
#a[2] = [[None]]*4
#a[2][0] = ones((4,4))
#a[2][1] = 2*ones((4,4))
#a[2][2] = 3*ones((4,4))
#a[2][3] = 4*ones((4,4))

#c,d = pdfb2vec(a)

# print c, c.shape, c.dtype
# print d, d.shape, d.dtype
