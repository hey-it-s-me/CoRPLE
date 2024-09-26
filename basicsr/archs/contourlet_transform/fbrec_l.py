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
from qprec import *
from pprec import *


def fbrec_l(y0, y1, f, type1, type2, extmod):
    """ FBREC_L   Two-channel 2D Filterbank Reconstruction
    using Ladder Structure

    x = fbrec_l(y0, y1, f, type1, type2, [extmod])

    Input:
    y0, y1:	two input subband images
    f:	filter in the ladder network structure
    type1:	'q' or 'p' for selecting quincunx or parallelogram
    downsampling matrix
    type2:	second parameter for selecting the filterbank type
    If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
    ({2, 3, 1, 4} can also be used as equivalent)
    If type1 == 'p' then type2 is one of {1, 2, 3, 4}
    Those are specified in QPDEC and PPDEC
    extmod: [optional] extension mode (default is 'per')
    This refers to polyphase components.

    Output:
    x:	reconstructed image

    Note:		This is also called the lifting scheme

    See also:	FBDEC_L"""

    # Modulate f
    f[2::] = -f[2::]

    if extmod is None:
        extmod = 'per'

    # Ladder network structure
    p1 = (-1 / sqrt(2)) * (y1 + sefilter2(y0, f, f, extmod))
    p0 = sqrt(2) * y0 + sefilter2(p1, f, f, extmod, [1, 1])

    # Polyphase reconstruction
    if lower(type1[0]) == 'q':
        # Quincunx polyphase reconstruction
        x = qprec(p0, p1, type2)
    elif lower(type1[0]) == 'p':
        # Parallelogram polyphase reconstruction
        x = pprec(p0, p1, type2)
    else:
        print 'Invalid argument type1'

    return x
