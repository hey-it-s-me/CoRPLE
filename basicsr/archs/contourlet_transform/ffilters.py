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
from .modulate2 import *


def ffilters(h0, h1):
    """ FFILTERS	Fan filters from diamond shape filters
    [f0, f1] = ffilters(h0, h1)"""

    f0 = [[None]] * 4
    f1 = [[None]] * 4

    # For the first half channels
    f0[0] = modulate2(h0, 'r', None)
    f1[0] = modulate2(h1, 'r', None)

    f0[1] = modulate2(h0, 'c', None)
    f1[1] = modulate2(h1, 'c', None)

    # For the second half channels,
    # use the transposed filters of the first half channels
    f0[2] = f0[0].conj().T
    f1[2] = f1[0].conj().T

    f0[3] = f0[1].conj().T
    f1[3] = f1[1].conj().T

    return f0, f1
