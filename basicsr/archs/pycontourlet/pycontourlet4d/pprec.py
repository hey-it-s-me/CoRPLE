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


def pprec(p0, p1, type):
    """% PPREC   Parallelogram Polyphase Reconstruction
    %
    % 	x = pprec(p0, p1, type)
    %
    % Input:
    %	p0, p1:	two parallelogram polyphase components of the image
    %	type:	one of {1, 2, 3, 4} for selecting sampling matrices:
    %			P1 = [2, 0; 1, 1]
    %			P2 = [2, 0; -1, 1]
    %			P3 = [1, 1; 0, 2]
    %			P4 = [1, -1; 0, 2]
    %
    % Output:
    %	x:	reconstructed image
    %
    % Note:
    %	These sampling matrices appear in the directional filterbank:
    %		P1 = R1 * Q1
    %		P2 = R2 * Q2
    %		P3 = R3 * Q2
    %		P4 = R4 * Q1
    %	where R's are resampling matrices and Q's are quincunx matrices
    %
    %	Also note that R1 * R2 = R3 * R4 = I so for example,
    %	upsample by R1 is the same with down sample by R2
    %
    % See also:	PPDEC"""

    # % Parallelogram polyphase decomposition by simplifying sampling matrices
    # % using the Smith decomposition of the quincunx matrices

    m, n = shape(p0)

    if type == 0:  # % P1 = R1 * Q1 = D1 * R3
        x = zeros((2 * m, n))
        x[::2, :] = resamp(p0, 3, None, None)
        temp = resamp(p1, 3, None, None)
        x[1::2, 1:] = temp[:, :-1:]
        x[1::2, 0:1] = temp[:, -1::]
    elif type == 1:  # % P2 = R2 * Q2 = D1 * R4
        x = zeros((2 * m, n))
        x[::2, :] = resamp(p0, 2, None, None)
        x[1::2, :] = resamp(p1, 2, None, None)
    elif type == 2:  # % P3 = R3 * Q2 = D2 * R1
        x = zeros((m, 2 * n))
        x[:, ::2] = resamp(p0, 1, None, None)
        temp = resamp(p1, 1, None, None)
        x[1:, 1::2] = temp[:-1:, ::]
        x[0:1, 1::2] = temp[-1::, ::]
    elif type == 3:  # % P4 = R4 * Q1 = D2 * R2
        x = zeros((m, 2 * n))
        x[:, ::2] = resamp(p0, 0, None, None)
        x[:, 1::2] = resamp(p1, 0, None, None)
    else:
        print 'Invalid argument type'
    return x
