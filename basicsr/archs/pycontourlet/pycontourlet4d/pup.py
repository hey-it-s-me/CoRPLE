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


def pup(x, type, phase):
    """ PUP   Parallelogram Upsampling

        y = pup(x, type, [phase])

     Input:
        x:	input image
        type:	one of {1, 2, 3, 4} for selecting sampling matrices:
                        P1 = [2, 0; 1, 1]
                        P2 = [2, 0; -1, 1]
                        P3 = [1, 1; 0, 2]
                        P4 = [1, -1; 0, 2]
        phase:	[optional] 0 or 1 to specify the phase of the input image as
                zero- or one-polyphase	component, (default is 0)

     Output:
        y:	parallelogram upsampled image

     Note:
        These sampling matrices appear in the directional filterbank:
                P1 = R1 * Q1
                P2 = R2 * Q2
                P3 = R3 * Q2
                P4 = R4 * Q1
        where R's are resampling matrices and Q's are quincunx matrices

     See also:	PPDEC"""

    if phase is None:
        phase = 0

    # Parallelogram polyphase decomposition by simplifying sampling matrices
    # using the Smith decomposition of the quincunx matrices
    #
    # Note that R1 * R2 = R3 * R4 = I so for example,
    # upsample by R1 is the same with down sample by R2.
    # Also the order of upsampling operations is in the reserved order
    # with the one of matrix multiplication.

    m, n = x.shape

    if type == 0:  # % P1 = R1 * Q1 = D1 * R3
        y = zeros((2 * m, n))
        if phase == 0:
            y[::2, :] = resamp(x, 3, None, None)
        else:
            temp = resamp(x, 3, None, None)
            y[1::2, 1:] = temp[:, :-1:]
            y[1::2, 0:1] = temp[:, -1::]
    elif type == 1:  # % P2 = R2 * Q2 = D1 * R4
        y = zeros((2 * m, n))
        if phase == 0:
            y[::2, :] = resamp(x, 2, None, None)
        else:
            y[1::2, :] = resamp(x, 2, None, None)
    elif type == 2:  # % P3 = R3 * Q2 = D2 * R1
        y = zeros((m, 2 * n))
        if phase == 0:
            y[:, ::2] = resamp(x, 1, None, None)
        else:
            temp = resamp(x, 1, None, None)
            y[1:, 1::2] = temp[:-1:, ::]
            y[0:1, 1::2] = temp[-1::, ::]
    elif type == 3:  # % P4 = R4 * Q1 = D2 * R2
        y = zeros((m, 2 * n))
        if phase == 0:
            y[:, ::2] = resamp(x, 0, None, None)
        else:
            y[:, 1::2] = resamp(x, 0, None, None)
    else:
        print "Invalid argument type"

    return y
