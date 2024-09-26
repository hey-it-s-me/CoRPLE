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

import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})
from .resampc import *


def resamp(x, type, shift, extmod):
    """ RESAMP   Resampling in 2D filterbank

        y = resamp(x, type, [shift, extmod])

        Input:
        x:	input image
        type: one of {0,1,2,3} (see note)

        shift:	[optional] amount of shift (default is 1)
        extmod: [optional] extension mode (default is 'per').
        Other options are:

        Output:
        y:	resampled image.

        Note:
        The resampling matrices are:
                R1 = [1, 1;  0, 1];
                R2 = [1, -1; 0, 1];
                R3 = [1, 0;  1, 1];
                R4 = [1, 0; -1, 1];

        For type 1 and type 2, the input image is extended (for example
        periodically) along the vertical direction;
        while for type 3 and type 4 the image is extended along the
        horizontal direction.

        Calling resamp(x, type, n) which n is positive integer is equivalent
        to repeatly calling resamp(x, type) n times.

        Input shift can be negative so that resamp(x, 1, -1) is the same
        with resamp(x, 2, 1)"""

    if shift is None:
        shift = 1

    if extmod is None:
        extmod = 'per'

    if type == 0 or type == 1:
        y = resampc(x, type, shift, extmod)
    elif type == 2 or type == 3:
        y = resampc(x.T, type - 2, shift, extmod).T
    else:
        print("The second input (type) must be one of {0, 1, 2, 3}")

    return y

#x  =  arange(1,26).reshape(5,5).T
#type = 4
#shift  = 1
#extmod = 'per'

#y = resamp(x, type, shift, extmod)

# print y
