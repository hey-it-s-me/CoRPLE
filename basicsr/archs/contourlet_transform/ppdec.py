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


def ppdec(x, type):
    """% PPDEC   Parallelogram Polyphase Decomposition
    %
    % 	[p0, p1] = ppdec(x, type)
    %
    % Input:
    %	x:	input image
    %	type:	one of {1, 2, 3, 4} for selecting sampling matrices:
    %			P1 = [2, 0; 1, 1]
    %			P2 = [2, 0; -1, 1]
    %			P3 = [1, 1; 0, 2]
    %			P4 = [1, -1; 0, 2]
    %
    % Output:
    %	p0, p1:	two parallelogram polyphase components of the image
    %
    % Note:
    %	These sampling matrices appear in the directional filterbank:
    %		P1 = R1 * Q1
    %		P2 = R2 * Q2
    %		P3 = R3 * Q2
    %		P4 = R4 * Q1
    %	where R's are resampling matrices and Q's are quincunx matrices
    %
    % See also:	QPDEC"""

    # % Parallelogram polyphase decomposition by simplifying sampling matrices
    # % using the Smith decomposition of the quincunx matrices

    if type == 0:  # % P1 = R1 * Q1 = D1 * R3
        p0 = resamp(x[::2, :], 2, None, None)
        # R1 * [0; 1] = [1; 1]
        p1 = resamp(hstack((x[1::2, 1:], x[1::2, 0:1])), 3, None, None)
    elif type == 1:  # P2 = R2 * Q2 = D1 * R4
        p0 = resamp(x[::2, :], 3, None, None)
        # R2 * [1; 0] = [1; 0]
        p1 = resamp(x[1::2, :], 3, None, None)
    elif type == 2:  # % P3 = R3 * Q2 = D2 * R1
        p0 = resamp(x[:, ::2], 0, None, None)
        # % R3 * [1; 0] = [1; 1]
        p1 = resamp(hstack((x[1:, 1::2].conj().T,
                            x[0:1, 1::2].conj().T)).conj().T, 0, None, None)
    elif type == 3:  # % P4 = R4 * Q1 = D2 * R2
        p0 = resamp(x[:, ::2], 1, None, None)
        # % R4 * [0; 1] = [0; 1]
        p1 = resamp(x[:, 1::2], 1, None, None)
    else:
        print 'Invalid argument type'
    return p0, p1
