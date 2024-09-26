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


def dfbimage(y, gap, gridI):
    """ DFBIMAGE    Produce an image from the result subbands of DFB

    im = dfbimage(y, [gap, gridI])

    Input:
    y:	output from DFBDEC
    gap:	gap (in pixels) between subbands
    gridI:	intensity of the grid that fills in the gap

    Output:
    im:	an image with all DFB subbands

    The subband images are positioned as follows
    (for the cases of 4 and 8 subbands):

    0   1              0   2
             and       1   3
    2   3            4 5 6 7

 History:
   09/17/2003  Creation.
   03/31/2004  Change the arrangement of the subbands to highlight
               the tree structure of directional partition """
    # Gap between subbands
    if gap is None:
        gap = 0

    l = len(y)

    # Intensity of the grid (default is white)
    if gridI is None:
        gridI = 0
        for k in xrange(0, l):
            m = abs(y[k]).max()
            # m = Inf;
            if m > gridI:
                gridI = m

    # gridI = gridI * 1.1;     # add extra 10% of intensity

    # Add grid seperation if required
    if gap > 0:
        for k in xrange(0, l):
            y[k][0:gap, :] = gridI
            y[k][:, 0:gap] = gridI

    # Simple case, only 2 subbands
    if l == 2:
        im = r_[y[0], y[1]]
        return im

    # Assume that the first subband has "horizontal" shape
    m, n = y[0].shape

    # The image
    im = zeros((l * m / 2, 2 * n))

    # First half of subband images ("horizontal" ones)
    for k in xrange(0, (l / 4)):
        im[arange(0, m) + k * m, :] = c_[y[k], y[(l / 4) + k]]

    # Second half of subband images ("vertical" ones)
    # The size of each of those subband
    # It must be that: p = l*m/4  and n = l*q/4
    p, q = y[l / 2 + 1].shape

    for k in xrange(0, (l / 2)):
        im[p::, arange(0, q) + k * q] = y[(l / 2) + k]

    # Finally, grid line in bottom and left
    # if gap > 0:
    # im(end-gap+1:end, :) = gridI
    # im(:, end-gap+1:end) = gridI

    return im
