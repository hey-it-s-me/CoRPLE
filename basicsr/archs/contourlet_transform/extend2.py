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


def extend2(x, ru, rd, cl, cr, extmod):
    """ EXTEND2   2D extension
    y = extend2(x, ru, rd, cl, cr, extmod)

    Input:
    x:	input image
    ru, rd:	amount of extension, up and down, for rows
    cl, cr:	amount of extension, left and rigth, for column
    extmod:	extension mode.  The valid modes are:
    'per':		periodized extension (both direction)
    'qper_row':	quincunx periodized extension in row
    'qper_col':	quincunx periodized extension in column

    Output:
    y:	extended image

    Note:
    Extension modes 'qper_row' and 'qper_col' are used multilevel
    quincunx filter banks, assuming the original image is periodic in
    both directions.  For example:
    [y0, y1] = fbdec(x, h0, h1, 'q', '1r', 'per');
    [y00, y01] = fbdec(y0, h0, h1, 'q', '2c', 'qper_col');
    [y10, y11] = fbdec(y1, h0, h1, 'q', '2c', 'qper_col');

    See also:	FBDEC"""

    rx, cx = array(x.shape)

    if extmod == 'per':

        I = getPerIndices(rx, ru, rd)
        y = x[I, :]

        I = getPerIndices(cx, cl, cr)
        y = y[:, I]
        return y
    elif extmod == 'qper_row':
        rx2 = round(rx / 2.0)
        y = c_[r_[x[rx2:rx, cx - cl:cx], x[0:rx2, cx - cl:cx]],
               x, r_[x[rx2:rx, 0:cr], x[0:rx2, 0:cr]]]
        I = getPerIndices(rx, ru, rd)
        y = y[I, :]
        return y
    elif extmod == 'qper_col':
        cx2 = int(round(cx / 2.0))
#         print("(extend2.py) c_[x[rx - ru:rx, cx2:cx], x[rx - ru:rx, 0:cx2]]", c_[x[rx - ru:rx, cx2:cx], x[rx - ru:rx, 0:cx2]])
        
        y = r_[c_[x[rx - ru:rx, cx2:cx], x[rx - ru:rx, 0:cx2]],
               x, c_[x[0:rd, cx2:cx], x[0:rd, 0:cx2]]]

        I = getPerIndices(cx, cl, cr)
        y = y[:, I]
        return y
    else:
        print("Invalid input for EXTMOD")

#----------------------------------------------------------------------------#
# Internal Function(s)
#----------------------------------------------------------------------------#


def getPerIndices(lx, lb, le):
    I = hstack((arange(lx - lb, lx), arange(0, lx), arange(0, le)))
    if (lx < lb) or (lx < le):
        I = mod(I, lx)
        I[I == 0] = lx
    return I.astype(int)

#x = arange(1,26).reshape(5,5)
#extmod = 'qper_row'
#a =  extend2(x,0,0,0,1,extmod)
# print a
