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


def ldfilter(fname):
    """LDFILTER	Generate filter for the ladder structure network
    f = ldfilter(fname)

    Input: fname:  Available 'fname' are:
    'pkvaN': length N filter from Phoong, Kim, Vaidyanathan and Ansari"""

    if fname == "pkva12" or fname == "pkva":
        v = array([0.6300, -0.1930, 0.0972, -0.0526, 0.0272, -0.0144])
    elif fname == "pkva8":
        v = array([0.6302, -0.1924, 0.0930, -0.0403])
    elif fname == "pkva6":
        v = array([0.6261, -0.1794, 0.0688])
    else:
        print("Unrecognized ladder structure filter name")
    # Symmetric impulse response
    f = hstack((v[::-1], v))
    return f
