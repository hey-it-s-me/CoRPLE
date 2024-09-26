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
from .ldfilter import *


def pfilters(fname):
    """ PFILTERS Generate filters for the laplacian pyramid

    Input:
    fname : Name of the filter, including the famous '9-7' filters.

    Output:
    h, g: 1D filters (lowpass for analysis and synthesis, respectively)
    for separable pyramid"""

    if fname == "9/7" or fname == "9-7":
        h = array([.037828455506995, -.023849465019380, -.11062440441842,
                   .37740285561265])
        h = hstack((h, .85269867900940, h[::-1]))

        g = array([-.064538882628938, -.040689417609558, .41809227322221])
        g = hstack((g, .78848561640566, g[::-1]))

        return h, g
    elif fname == "maxflat":
        M1 = 1 / sqrt(2)
        M2 = M1
        k1 = 1 - sqrt(2)
        k2 = M1
        k3 = k1
        h = array([.25 * k2 * k3, .5 * k2, 1 + .5 * k2 * k3]) * M1
        h = hstack((h, h[len(h) - 2::-1]))

        g = array([-.125 * k1 * k2 * k3, 0.25 * k1 * k2, -0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3,
                   1 + .5 * k1 * k2]) * M2
        g = hstack((g, g[len(g) - 2::-1]))
        # Normalize
        h = h * sqrt(2)
        g = g * sqrt(2)
        return h, g
    elif fname == "5/3" or fname == "5-3":
        h = [-1, 2, 6, 2, -1] / (4 * sqrt(2))
        g = [1, 2, 1] / (2 * sqrt(2))
        return h, g
    elif fname == "burt" or fname == "Burt":
        h = array([0.6, 0.25, -0.05])
        h = sqrt(2) * hstack((h[len(h):0:-1], h))

        g = array([17.0 / 28, 73.0 / 280, -3.0 / 56, -3.0 / 280])
        g = sqrt(2) * hstack((g[len(g):0:-1], g))
        return h, g
    elif fname == "pkva":
        # filters from the ladder structure
        # Allpass filter for the ladder structure network
        beta = ldfilter(fname)

        lf = len(beta)
        n = float(lf) / 2

        if n != floor(n):
            print("The input allpass filter must be even length")

        # beta(z^2)
        beta2 = zeros(2 * lf - 1)
        beta2[::2] = beta

        # H(z)
        h = beta2.copy()
        h[2 * n - 1] = h[2 * n - 1] + 1
        h = h / 2

        # G(z)
        g = -convolve(beta2, h)
        g[4 * n - 2] = g[4 * n - 2] + 1
        g[1:-1:2] = -g[1:-1:2]

        # Normalize
        h = h * sqrt(2)
        g = g * sqrt(2)
        return h, g
#h,g = pfilters("pkva")
# print h,g
