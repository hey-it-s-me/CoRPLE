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


def qdown(x, type, extmod, phase):
    """% QDOWN   Quincunx Downsampling
    %
    % 	y = qdown(x, [type], [extmod], [phase])
    %
    % Input:
    %	x:	input image
    %	type:	[optional] one of {'1r', '1c', '2r', '2c'} (default is '1r')
    %		'1' or '2' for selecting the quincunx matrices:
    %			Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
    %		'r' or 'c' for suppresing row or column
    %	phase:	[optional] 0 or 1 for keeping the zero- or one-polyphase
    %		component, (default is 0)
    %
    % Output:
    %	y:	qunincunx downsampled image
    %
    % See also:	QPDEC"""

    if type is None:
        type = '1r'

    if phase is None:
        phase = 0

    """% Quincunx downsampling using the Smith decomposition:
    %	Q1 = R2 * [2, 0; 0, 1] * R3
    %	   = R3 * [1, 0; 0, 2] * R2
    % and,
    %	Q2 = R1 * [2, 0; 0, 1] * R4
    %	   = R4 * [1, 0; 0, 2] * R1
    %
    % See RESAMP for the definition of those resampling matrices"""

    if type == '1r':
        z = resamp(x, 1, None, None)
        if phase == 0:
            y = resamp(z[::2, :], 2, None, None)
        else:
            y = resamp(hstack((z[1::2, 1:], z[1::2, 0:1])), 2, None, None)

    elif type == '1c':
        z = resamp(x, 2, None, None)
        if phase == 0:
            y = resamp(z[:, ::2], 1, None, None)
        else:
            y = resamp(z[:, 1::2], 1, None, None)
    elif type == '2r':
        z = resamp(x, 0, None, None)
        if phase == 0:
            y = resamp(z[::2, :], 3, None, None)
        else:
            y = resamp(z[1::2, :], 3, None, None)
    elif type == '2c':
        z = resamp(x, 3, None, None)
        if phase == 0:
            y = resamp(z[:, ::2], 0, None, None)
        else:
            y = resamp(hstack((z[1:, 1::2].conj().T,
                               z[0:1, 1::2].conj().T)).conj().T, 0, None, None)
    else:
        print("Invalid argument type")
    return y
