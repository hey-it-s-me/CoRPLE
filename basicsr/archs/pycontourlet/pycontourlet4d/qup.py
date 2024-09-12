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


def qup(x, type, phase):
    """% QUP   Quincunx Upsampling
    %
    % 	y = qup(x, [type], [phase])
    %
    % Input:
    %	x:	input image
    %	type:	[optional] one of {'1r', '1c', '2r', '2c'} (default is '1r')
    %		'1' or '2' for selecting the quincunx matrices:
    %			Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
    %		'r' or 'c' for extending row or column
    %	phase:	[optional] 0 or 1 to specify the phase of the input image as
    %		zero- or one-polyphase component, (default is 0)
    %
    % Output:
    %	y:	qunincunx upsampled image
    %
    % See also:	QDOWN"""

    if type is None:
        type = "1r"

    if phase is None:
        phase = 0

    """% Quincunx downsampling using the Smith decomposition:
    %
    %	Q1 = R2 * [2, 0; 0, 1] * R3
    %	   = R3 * [1, 0; 0, 2] * R2
    % and,
    %	Q2 = R1 * [2, 0; 0, 1] * R4
    %	   = R4 * [1, 0; 0, 2] * R1
    %
    % See RESAMP for the definition of those resampling matrices
    %
    % Note that R1 * R2 = R3 * R4 = I so for example,
    % upsample by R1 is the same with down sample by R2.
    % Also the order of upsampling operations is in the reserved order
    % with the one of matrix multiplication."""

    m, n = shape(x)
    if type == "1r":
        z = zeros((2 * m, n))
        if phase == 0:
            z[::2, :] = resamp(x, 3, None, None)
        else:
            temp = resamp(x, 3, None, None)
            z[1::2, 1:] = temp[:, :-1:]
            z[1::2, 0:1] = temp[:, -1::]
        y = resamp(z, 0, None, None)
    elif type == "1c":
        z = zeros((m, 2 * n))
        if phase == 0:
            z[:, ::2] = resamp(x, 0, None, None)
        else:
            z[:, 1::2] = resamp(x, 0, None, None)
        y = resamp(z, 3, None, None)
    elif type == "2r":
        z = zeros((2 * m, n))
        if phase == 0:
            z[::2, :] = resamp(x, 2, None, None)
        else:
            z[1::2, :] = resamp(x, 2, None, None)
        y = resamp(z, 1, None, None)
    elif type == '2c':
        z = zeros((m, 2 * n))
        if phase == 0:
            z[:, ::2] = resamp(x, 1, None, None)
        else:
            temp = resamp(x, 1, None, None)
            z[1:, 1::2] = temp[:-1:, ::]
            z[0:1, 1::2] = temp[-1::, ::]
        y = resamp(z, 2, None, None)
    else:
        print("Invalid argument type")

    return y
