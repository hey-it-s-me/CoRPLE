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


def pdfbrec(y, pfilt, dfilt)
    """% PDFBREC   Pyramid Directional Filterbank Reconstruction
    %
    %	x = pdfbrec(y, pfilt, dfilt)
    %
    % Input:
    %   y:	    a cell vector of length n+1, one for each layer of
    %       	subband images from DFB, y{1} is the low band image
    %   pfilt:  filter name for the pyramid
    %   dfilt:  filter name for the directional filter bank
    %
    % Output:
    %   x:      reconstructed image
    %
    % See also: PFILTERS, DFILTERS, PDFBDEC"""

    n = len(y) - 1
    if n <= 0:
        x = y[0]
    else:
        'Recursive call to reconstruct the low band
        xlo = pdfbrec(y[0:-1], pfilt, dfilt)
        'Get the pyramidal filters from the filter name
        h, g = pfilters(pfilt)
        'Process the detail subbands
        if len(y[-1]) != 3:
            ' Reconstruct the bandpass image from DFB
            ' Decide the method based on the filter name

            if dfilt == 'pkva6' or dfilt == 'pkva8' or dfilt == 'pkva12' or dfilt == 'pkva':
                ' Use the ladder structure(much more efficient)
                xhi = dfbrec_l(y[-1], dfilt)
            else:
                ' General case
                xhi = dfbrec(y[-1], dfilt)
            x = lprec(xlo, xhi, h, g)
        else:
            ' Special case: length(y{end}) == 3
            ' Perform one - level 2 - D critically sampled wavelet filter bank
            x = wfb2rec(xlo, y[-1][0], y[-1][1], y[-1][2], h, g)
    return x
