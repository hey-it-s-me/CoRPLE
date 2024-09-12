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
from numpy.linalg import norm


def resampz(x, type, shift):
    """ RESAMPZ   Resampling of matrix
        y = resampz(x, type, [shift])

        Input:
        x:      input matrix
        type:   one of {0, 1, 2, 3} (see note)
        shift:  [optional] amount of shift (default is 1)

        Output:
        y:      resampled matrix

        Note:
        The resampling matrices are:
                R1 = [1, 1;  0, 1];
                R2 = [1, -1; 0, 1];
                R3 = [1, 0;  1, 1];
                R4 = [1, 0; -1, 1];

        This resampling program does NOT involve periodicity, thus it
        zero-pad and extend the matrix."""
    if shift is None:
        shift = 1

    sx = array(x.shape)

    if type == 0 or type == 1:
        y = zeros([sx[0] + abs(shift * (sx[1] - 1)), sx[1]])

        if type == 0:
            shift1 = arange(sx[1]) * (-shift)
        else:
            shift1 = arange(sx[1]) * shift

        # Adjust to non-negative shift if needed
        if shift1[-1] < 0:
            shift1 = shift1 - shift1[-1]

        for n in xrange(sx[1]):
            y[shift1[n] + arange(sx[0]), n] = x[:, n].copy()

        # Finally, delete zero rows if needed
        start = 0
        finish = array(y.shape[0])

        while norm(y[start, :]) == 0:
            start = start + 1

        while norm(y[finish - 1, :]) == 0:
            finish = finish - 1

        y = y[start:finish, :]
    elif type == 2 or type == 3:
        y = zeros([sx[0], sx[1] + abs(shift * (sx[0] - 1))])
        if type == 2:
            shift2 = arange(sx[0]) * (-shift)
        else:
            shift2 = arange(sx[0]) * shift

        # Adjust to non-negative shift if needed
        if shift2[-1] < 0:
            shift2 = shift2 - shift2[-1]

        for m in xrange(sx[0]):
            y[m, shift2[m] + arange(sx[1])] = x[m, :].copy()

        # Finally, delete zero columns if needed
        start = 0
        finish = array(y.shape[1])

        while norm(y[:, start]) == 0:
            start = start + 1

        while norm(y[:, finish - 1]) == 0:
            finish = finish - 1
        y = y[:, start:finish]

    return y
