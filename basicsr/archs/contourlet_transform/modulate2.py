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


def modulate2(x, type_, center):
    """ MODULATE2 2D modulation
    y = modulate2(x, type, [center])

    With TYPE = {'r', 'c' or 'b'} for modulate along the row, or column or
    both directions.
    CENTER especify the origin of modulation as
    floor(size(x)/2)+center(default is [0, 0])"""

    if center is None:
        center = array([[0, 0]])

    # Size and origin
    if x.ndim > 1:
        s = array([x.shape])
    else:
        x = array([x])
        s = array(x.shape)

    #o = floor(s / 2) + 1 + center
    o = floor(s / 2.0) + center
    n1 = array([s[0][0]]) - o[0][0]
    n2 = array([s[0][1]]) - o[0][1]

    if str.lower(type_[0]) == 'r':
        m1 = (-1)**n1
        y = x * tile(m1.conj().T, s[0][1])

    elif str.lower(type_[0]) == 'c':
        m2 = (-1)**n2
        y = x * tile(m2, s[0])

    elif str.lower(type_[0]) == 'b':
        m1 = (-1)**n1
        m2 = (-1)**n2
        m = m1.conj().T * m2
        y = x * m

    return y
