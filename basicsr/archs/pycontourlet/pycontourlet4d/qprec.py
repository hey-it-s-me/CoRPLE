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


def qprec(p0, p1, type):
    """% QPREC   Quincunx Polyphase Reconstruction
    %
    % 	x = qprec(p0, p1, [type])
    %
    % Input:
    %	p0, p1:	two qunincunx polyphase components of the image
    %	type:	[optional] one of {'1r', '1c', '2r', '2c'}, default is '1r'
    %		'1' and '2' for selecting the quincunx matrices:
    %			Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
    %		'r' and 'c' for suppresing row or column
    %
    % Output:
    %	x:	reconstructed image
    %
    % Note:
    %	Note that R1 * R2 = R3 * R4 = I so for example,
    %	upsample by R1 is the same with down sample by R2
    %
    % See also:	QPDEC"""
    if type is None:
        type = "1r"

    """% Quincunx downsampling using the Smith decomposition:
    %
    %       Q1 = R2 * D1 * R3
    %          = R3 * D2 * R2
    % and,
    %       Q2 = R1 * D1 * R4
    %          = R4 * D2 * R1
    %
    % where D1 = [2, 0; 0, 1] and D2 = [1, 0; 0, 2].
    % See RESAMP for the definition of the resampling matrices R's"""

    m, n = shape(p0)

    if type == "1r":  # % Q1 = R2 * D1 * R3
        y = zeros((2 * m, n))
        y[::2, :] = resamp(p0, 3, None, None)
        temp = resamp(p1, 3, None, None)
        y[1::2, 1:] = temp[:, :-1:]
        y[1::2, 0:1] = temp[:, -1::]
        x = resamp(y, 0, None, None)

    elif type == "1c":  # % Q1 = R3 * D2 * R2
        y = zeros((m, 2 * n))
        y[:, ::2] = resamp(p0, 0, None, None)
        y[:, 1::2] = resamp(p1, 0, None, None)
        x = resamp(y, 3, None, None)
    elif type == "2r":  # % Q2 = R1 * D1 * R4
        y = zeros((2 * m, n))
        y[::2, :] = resamp(p0, 2, None, None)
        y[1::2, :] = resamp(p1, 2, None, None)
        x = resamp(y, 1, None, None)
    elif type == "2c":  # % Q2 = R4 * D2 * R1
        y = zeros((m, 2 * n))
        y[:, ::2] = resamp(p0, 1, None, None)
        temp = resamp(p1, 1, None, None)
        y[1:, 1::2] = temp[:-1:, ::]
        y[0:1, 1::2] = temp[-1::, ::]
        x = resamp(y, 2, None, None)
    else:
        print 'Invalid argument type'
    return x
