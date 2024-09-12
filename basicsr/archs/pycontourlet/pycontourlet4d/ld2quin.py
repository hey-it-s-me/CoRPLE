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
from .qupz import *
from .ldfilter import *


def ld2quin(beta):
    """LD2QUIN    Quincunx filters from the ladder network structure
    Construct the quincunx filters from an allpass filter (beta) using the
    ladder network structure
    Ref: Phong et al., IEEE Trans. on SP, March 1995"""

    if beta.ndim > 1:
        raise ValueError("The input must be an 1-D filter")

    # Make sure beta is a row vector
    beta = beta[:, newaxis].reshape(len(beta),)

    lf = len(beta)
    n = lf / 2.0

    if n != floor(n):
        raise ValueError("The input allpass filter must be even length")

    # beta(z1) * beta(z2)
    sp = beta.conj().T * beta

    # beta(z1*z2^{-1}) * beta(z1*z2)
    # Obtained by quincunx upsampling type 1 (with zero padded)
    h = qupz(sp, 1)

    # Lowpass quincunx filter
    h0 = h.copy()
    h0[2 * n, 2 * n] = h0[2 * n, 2 * n] + 1
    h0 = h0 / 2.0

    # Highpass quincunx filter
    h1 = -signal.convolve(h, h0)
    h1[4 * n - 1, 4 * n - 1] = h1[4 * n - 1, 4 * n - 1] + 1

    return h0, h1
