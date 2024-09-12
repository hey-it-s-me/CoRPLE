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


def wfb2rec(x_LL, x_LH, x_HL, x_HH, h, g):
    """% WFB2REC   2-D Wavelet Filter Bank Decomposition
    %
    %       x = wfb2rec(x_LL, x_LH, x_HL, x_HH, h, g)
    %
    % Input:
    %   x_LL, x_LH, x_HL, x_HH:   Four 2-D wavelet subbands
    %   h, g:   lowpass analysis and synthesis wavelet filters
    %
    % Output:
    %   x:      reconstructed image"""

    # Make sure filter in a row vector
    h = h[:, newaxis].reshape(len(h),)
    g = g[:, newaxis].reshape(len(g),)

    g0 = g.copy()
    len_g0 = len(g0)
    ext_g0 = floor((len_g0 - 1) / 2)

    # Highpass synthesis filter: G1(z) = -z H0(-z)
    len_g1 = len(h)
    c = floor((len_g1 + 1) / 2)
    g1 = (-1) * h * (-1) ** (arange(1, len_g1 + 1) - c)
    ext_g1 = len_g1 - (c + 1)

    # Get the output image size
    height, width = shape(x_LL)
    x_B = zeros((height * 2, width))
    x_B[::2, :] = x_LL

    # Column-wise filtering
    x_L = rowfiltering(x_B.conj().T, g0, ext_g0).conj().T
    x_B[::2, :] = x_LH
    x_L = x_L + rowfiltering(x_B.conj().T, g1, ext_g1).conj().T

    x_B[::2, :] = x_HL
    x_H = rowfiltering(x_B.conj().T, g0, ext_g0).conj().T
    x_B[::2, :] = x_HH
    x_H = x_H + rowfiltering(x_B.conj().T, g1, ext_g1).conj().T

    # Row-wise filtering
    x_B = zeros((2 * height, 2 * width))
    x_B[:, ::2] = x_L
    x = rowfiltering(x_B, g0, ext_g0)
    x_B[:, ::2] = x_H
    x = x + rowfiltering(x_B, g1, ext_g1)

    return x


# Internal function: Row-wise filtering with border handling
def rowfiltering(x, f, ext1):
    ext2 = len(f) - ext1 - 1
    x = hstack((x[:, -ext1::], x, x[:, 0:ext2]))
    y = signal.convolve(x.conj().T, f[:, newaxis], 'valid').conj().T
    return y

# x=ones((5,5))
# h=array([1.,2.,3.,4.])
# g=array([4.,3.,2.,1.])
# x_LL=100*ones((3,3))
# x_LH=-20*ones((3,3))
# x_HL=-20*ones((3,3))
# x_HH=4*ones((3,3))
#t=wfb2rec(x_LL, x_LH, x_HL, x_HH, h, g)
# print t
