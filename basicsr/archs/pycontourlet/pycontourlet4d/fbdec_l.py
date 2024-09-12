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
from qpdec import *
from ppdec import *
from sefilter2 import *


def fbdec_l(x, f, type1, type2, extmod):
    """ FBDEC_L   Two-channel 2D Filterbank Decomposition using Ladder Structure

    [y0, y1] = fbdec_l(x, f, type1, type2, [extmod])

    Input:
    x:	input image
    f:	filter in the ladder network structure
    type1:	'q' or 'p' for selecting quincunx or parallelogram
    downsampling matrix
    type2:	second parameter for selecting the filterbank type
    If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
    ({1, 2, 0, 3} can also be used as equivalent)
    If type1 == 'p' then type2 is one of {0, 1, 2, 3}
    Those are specified in QPDEC and PPDEC
    extmod:	[optional] extension mode (default is 'per')
    This refers to polyphase components.

    Output:
    y0, y1:	two result subband images

    Note:		This is also called the lifting scheme

    See also:	FBDEC, FBREC_L"""

    # Modulate f
    f[2::] = -f[2::]

    if min(x.shape) == 1:
        print 'Input is a vector, unpredicted output!'

    if extmod is None:
        extmod = 'per'

    # Polyphase decomposition of the input image
    if str.lower(type1[0]) == 'q':
        # Quincunx polyphase decomposition
        p0, p1 = qpdec(x, type2)
    elif str.lower(type1[0]) == 'p':
        # Parallelogram polyphase decomposition
        p0, p1 = ppdec(x, type2)

    else:
        print 'Invalid argument type1'

    # Ladder network structure
    y0 = (1 / sqrt(2)) * (p0 - sefilter2(p1, f, f, extmod, array([1, 1])))
    y1 = (-sqrt(2) * p1) - sefilter2(y0, f, f, extmod)

    return y0, y1
