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
from scipy import signal
#from scipy.signal.filter_design import firwin
from scipy.fftpack import fftshift
#from modulate2 import *


def mctrans(b, t):
    """ MCTRANS McClellan transformation
    H = mctrans(B,T) produces the 2-D FIR filter H that
    corresponds to the 1-D FIR filter B using the transform T."""

    # Convert the 1-D filter b to SUM_n a(n) cos(wn) form
    #n = (len(b)-1) / 2
    n = int((len(b) - 1) / 2.0)
    if b.ndim < 2:
        b = fftshift(b[::-1])[::-1]  # inverse fftshift
    else:
        b = rot90(fftshift(rot90(b, 2)), 2)  # inverse fftshift

    a = hstack((b[0], 2 * b[1:n + 1]))

    inset = floor((array(t.shape) - 1) / 2).astype(int)

    # Use Chebyshev polynomials to compute h
    P0, P1 = 1, t.copy()
    h = a[1] * P1
    rows, cols = array([inset[0]]), array([inset[1]])
    h[rows, cols] = h[rows, cols] + a[0] * P0

    for i in range(2, n + 1):
        P2 = 2 * signal.convolve(t, P1)
        rows = rows + inset[0]
        cols = cols + inset[1]
        if rows.shape[0] > 1:
            P2[ix_(rows, cols)] = P2[ix_(rows, cols)] - P0
        else:
            P2[rows, cols] = P2[rows, cols] - P0
        rows = inset[0] + arange(0, P1.shape[0])
        cols = inset[1] + arange(0, P1.shape[1])
        hh = h.copy()
        h = a[i] * P2
        h[ix_(rows, cols)] = h[ix_(rows, cols)] + hh
        P0, P1 = P1.copy(), P2.copy()

    h = rot90(h, 2)  # Rotate for use with filter2
    return h

#b = firwin(5,0.5)
#t = array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0
# h0 = array([0.026748757411, -0.016864118443, -0.078223266529,
 #                   0.266864118443, 0.602949018236, 0.266864118443,
  #                  -0.078223266529, -0.016864118443, 0.026748757411])
#h1 = modulate2(h0, 'r', None)
# print mctrans(h1.T,t)
# print h1, h1.shape,h0.shape
