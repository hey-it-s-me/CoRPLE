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
from resamp import *


def qpdec(x, type):
    """ QPDEC   Quincunx Polyphase Decomposition

        [p0, p1] = qpdec(x, [type])

     Input:
        x:	input image
        type:	[optional] one of {'1r', '1c', '2r', '2c'} default is '1r'
                '1' and '2' for selecting the quincunx matrices:
                        Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
                'r' and 'c' for suppresing row or column

     Output:
        p0, p1:	two qunincunx polyphase components of the image"""

    if type is None:
        type = "1r"

    """ Quincunx downsampling using the Smith decomposition:
    	Q1 = R2 * D1 * R3
    	   = R3 * D2 * R2
     and,
    	Q2 = R1 * D1 * R4
    	   = R4 * D2 * R1

     where D1 = [2, 0; 0, 1] and D2 = [1, 0; 0, 2].
     See RESAMP for the definition of the resampling matrices R's"""

    if type == "1r":  # Q1 = R2 * D1 * R3
        y = resamp(x, 1, None, None)
        p0 = resamp(y[::2, :], 2, None, None)
        # inv(R2) * [0; 1] = [1; 1]
        p1 = resamp(hstack((y[1::2, 1:], y[1::2, 0:1])), 2, None, None)

    elif type == "1c":      # Q1 = R3 * D2 * R2
        y = resamp(x, 2, None, None)
        p0 = resamp(y[:, ::2], 1, None, None)
        # inv(R3) * [0; 1] = [0; 1]
        p1 = resamp(y[:, 1::2], 1, None, None)
    elif type == "2r":  # Q2 = R1 * D1 * R4
        y = resamp(x, 0, None, None)
        p0 = resamp(y[::2, :], 3, None, None)
        # inv(R1) * [1; 0] = [1; 0]
        p1 = resamp(y[1::2, :], 3, None, None)
    elif type == "2c":  # Q2 = R4 * D2 * R1
        y = resamp(x, 3, None, None)
        p0 = resamp(y[:, ::2], 0, None, None)
        # inv(R4) * [1; 0] = [1; 1]
        p1 = resamp(hstack((y[1:, 1::2].conj().T,
                            y[0:1, 1::2].conj().T)).conj().T, 0, None, None)
    else:
        print 'Invalid argument type'

    return p0, p1
