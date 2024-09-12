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
from .resamp import *


def pdown(x, type, phase):
    """ PDOWN   Parallelogram Downsampling
        y = pdown(x, type, [phase])
     Input:
        x:	input image
        type:	one of {0, 1, 2, 3} for selecting sampling matrices:
                        P1 = [2, 0; 1, 1]
                        P2 = [2, 0; -1, 1]
                        P3 = [1, 1; 0, 2]
                        P4 = [1, -1; 0, 2]
        phase:	[optional] 0 or 1 for keeping the zero- or one-polyphase
                component, (default is 0)
     Output:
        y:	parallelogram downsampled image
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

    if type == 0:  # P1 = R1 * Q1 = D1 * R3
        if phase == 0:
            y = resamp(x[::2, :], 2, None, None)
        else:
            y = resamp(hstack((x[1::2, 1:], x[1::2, 0:1])), 2, None, None)
    elif type == 1:  # P2 = R2 * Q2 = D1 * R4
        if phase == 0:
            y = resamp(x[::2, :], 3, None, None)
        else:
            y = resamp(x[1::2, :], 3, None, None)
    elif type == 2:  # P3 = R3 * Q2 = D2 * R1
        if phase == 0:
            y = resamp(x[:, ::2], 0, None, None)
        else:
            y = resamp(hstack((x[1:, 1::2].conj().T,
                               x[0:1, 1::2].conj().T)).conj().T, 0, None, None)
    elif type == 3:  # P4 = R4 * Q1 = D2 * R2
        if phase == 0:
            y = resamp(x[:, ::2], 1, None, None)
        else:
            y = resamp(x[:, 1::2], 1, None, None)
    else:
        print("Invalid argument type")

    return y
