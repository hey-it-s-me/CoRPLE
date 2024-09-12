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


def wfb2dec(x, h, g):
    """% WFB2DEC   2-D Wavelet Filter Bank Decomposition
    %
    %       y = wfb2dec(x, h, g)
    %
    % Input:
    %   x:      input image
    %   h, g:   lowpass analysis and synthesis wavelet filters
    %
    % Output:
    %   x_LL, x_LH, x_HL, x_HH:   Four 2-D wavelet subbands"""

    # Make sure filter in a row vector
    h = h[:, newaxis].reshape(len(h),)
    g = g[:, newaxis].reshape(len(g),)

    h0 = h
    len_h0 = len(h0)
    ext_h0 = floor(len_h0 / 2.0)
    # Highpass analysis filter: H1(z) = -z^(-1) G0(-z)
    len_h1 = len(g)
    c = floor((len_h1 + 1.0) / 2.0)
    # Shift the center of the filter by 1 if its length is even.
    if mod(len_h1, 2) == 0:
        c = c + 1
    # print(c)
    h1 = - g * (-1)**(arange(1, len_h1 + 1) - c)
    ext_h1 = len_h1 - c + 1

    # Row-wise filtering
    x_L = rowfiltering(x, h0, ext_h0)
    x_L = x_L[:, ::2]  # (:, 1:2:end)

    x_H = rowfiltering(x, h1, ext_h1)
    x_H = x_H[:, ::2]  # x_H(:, 1:2:end);

    # Column-wise filtering
    x_LL = rowfiltering(x_L.conj().T, h0, ext_h0)
    x_LL = x_LL.conj().T
    x_LL = x_LL[::2, :]

    x_LH = rowfiltering(x_L.conj().T, h1, ext_h1)
    x_LH = x_LH.conj().T
    x_LH = x_LH[::2, :]

    x_HL = rowfiltering(x_H.conj().T, h0, ext_h0)
    x_HL = x_HL.conj().T
    x_HL = x_HL[::2, :]

    x_HH = rowfiltering(x_H.conj().T, h1, ext_h1)
    x_HH = x_HH.conj().T
    x_HH = x_HH[::2, :]

    return x_LL, x_LH, x_HL, x_HH

# Internal function: Row-wise filtering with border handling


def rowfiltering(x, f, ext1):
    ext1 = int(ext1)
    ext2 = int(len(f) - ext1 - 1)
    x = hstack((x[:, -ext1::], x, x[:, 0:ext2]))
    y = signal.convolve(x.conj().T, f[:, newaxis], 'valid').conj().T
    return y


# x=ones((5,5))
# h=array([1.,2.,3.,4.])
# g=array([4.,3.,2.,1.])
#x_LL, x_LH, x_HL, x_HH=wfb2dec(x, h, g)
# print x_LL, x_LH, x_HL, x_HH
