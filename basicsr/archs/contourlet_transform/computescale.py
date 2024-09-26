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


def computescale(cellDFB, dRatio, nStart, nEnd, coefMode):
    """
    COMPUTESCALE   Comupute display scale for PDFB coefficients

    computescale(cellDFB, [dRatio, nStart, nEnd, coefMode])

    Input:
    cellDFB:	a cell vector, one for each layer of
    subband images from DFB.
    dRatio:
    display ratio. It ranges from 1.2 to 10.

    nStart:
    starting index of the cell vector cellDFB for the computation.
    Its default value is 1.
    nEnd:
    ending index of the cell vector cellDFB for the computation.
    Its default value is the length of cellDFB.
    coefMode:
    coefficients mode (a string):
    'real' ----  Highpass filters use the real coefficients.
    'abs' ------ Highpass filters use the absolute coefficients.
    It's the default value
    Output:
    vScales ---- 1 X 2 vectors for two scales.

    History:
   10/03/2003  Creation.
   04/01/2004  Limit the display scale into the range of
   [min(celldfb), max(celldfb)] or [min(abs(celldfb)), max(abs(celldfb))]

   See also:     SHOWPDFB"""

    if ~iscell(cellDFB):
        print 'Error in computescale.py!
        The first input must be a cell vector'

    # Display ratio
    if dRatio is None:
        dRatio = 2
    elif dRatio < 1:
        print 'Warning! the display ratio must be larger than 1!
        Its defualt value is 2!'
    # Starting index for the cell vector cellDFB
    if nStart is None:
        nStart = 1
    elif nStart < 1 or nStart > len(cellDFB):
        print 'Warning! The starting index from 1 to length(cellDFB)!
        Its defualt value is 1!'
        nStart = 1

    # Starting index for the cell vector cellDFB
    if nEnd is None:
        nEnd = len(cellDFB)
    elif nEnd < 1 or nEnd > len(cellDFB):
        print 'Warning! The ending index from 1 to length(cellDFB)!
        Its defualt value is length(cellDFB)!'
        nEnd = len(cellDFB)
    # Coefficient mode
    if coefMode is None:
        coefMode = 'abs'
    elif coefMode != 'real' and coefMode != 'abs':
        print 'Warning! There are only two coefficients mode: real, abs!
        Its defualt value is "abs"!'
        coefMode = 'abs'

    # Initialization
    dSum = 0
    dMean = 0
    # Added on 04/01/04 by jpzhou
    dMin = 1.0e14
    dMax = -1.0e14
    dAbsMin = 1.0e14
    dAbsMax = -1.0e14
    dAbsSum = 0
    nCount = 0
    vScales = zeros((1, 2))

    if coefMode == 'real':  # Use the real coefficients
        # Compute the mean of all coefficients
        for i in xrange(nStart, nEnd):
            if iscell(cellDFB[i]):  # Check whether it is a cell vector
                m = len(cellDFB[i])
                for j in xrange(0, m):

                    # Added on 04/01/04 by jpzhou
                    dUnitMin = cellDFB[i][j].min()
                    if dUnitMin < dMin:
                        dMin = dUnitMin
                        dUnitMax = cellDFB[i][j].max()
                        if dUnitMax > dMax:
                            dMax = dUnitMax
                            dSum = dSum + sum(cellDFB[i][j])
                            nCount = nCount + cellDFB[i][j].size
            else:
                # Added on 04/01/04 by jpzhou
                dUnitMin = cellDFB[i].min()
                if dUnitMin < dMin:
                    dMin = dUnitMin
                    dUnitMax = cellDFB[i].max()
                    if dUnitMax > dMax:
                        dMax = dUnitMax
                        dSum = dSum + sum(cellDFB[i])
                        nCount = nCount + cellDFB[i].size
        if nCount < 2 or abs(dSum) < 1e-10:
            print 'Error in computescale.m! No data in this unit!'
        else:
            dMean = dSum / nCount

        # Compute the STD.
        dSum = 0
        for i in xrange(nStart, nEnd):
            if iscell(cellDFB[i]):  # Check whether it is a cell vector
                m = len(cellDFB[i])
                for j in xrange(0, m):
                    dSum = dSum + sum((cellDFB[i][j] - dMean)**2)
                    #nCount = nCount + cellDFB[i][j].size
            else:
                dSum = dSum + sum((cellDFB[i] - dMean)**2)
                # nCount = nCount + prod( size( cellDFB{i} ) )

        dStd = sqrt(dSum / (nCount - 1))

        # Modified on 04/01/04
        #dMin = -1.0e10 ;
        #dMax = 1.0e10 ;
        vScales[0] = max(dMean - dRatio * dStd, dMin)
        vScales[1] = min(dMean + dRatio * dStd, dMax)

    else:  # Use the absolute coefficients
        # Compute the mean of absolute values
        for i in xrange(nStart, nEnd):
            if iscell(cellDFB[i]):  # Check whether it is a cell vector
                m = len(cellDFB{i})
                for j in xrange(0, m):
                    # Added on 04/01/04 by jpzhou
                    dUnitMin = abs(cellDFB[i][j]).min()
                    if dUnitMin < dAbsMin:
                        dAbsMin = dUnitMin
                    dUnitMax = abs(cellDFB[i][j]).max()
                    if dUnitMax > dAbsMax:
                        dAbsMax = dUnitMax
                dAbsSum = dAbsSum + sum(abs(cellDFB[i][j]))
                nCount = nCount + cellDFB[i][j].size
            else:
                # Added on 04/01/04 by jpzhou
                dUnitMin = abs(cellDFB[i]).min()
            if dUnitMin < dAbsMin:
                dAbsMin = dUnitMin
            dUnitMax = abs(cellDFB[i]).max()
            if dUnitMax > dAbsMax:
                dAbsMax = dUnitMax
            dAbsSum = dAbsSum + sum(abs(cellDFB[i]))
            nCount = nCount + cellDFB[i].size
        if nCount < 2 or dAbsSum < 1e-10:
            'Error in computescale! No data in this unit!'
        else:
            dAbsMean = dAbsSum / nCount
        # Compute the std of absolute values
        dSum = 0
        for i in xrange(nStart, nEnd):
            if iscell(cellDFB[i]):  # Check whether it is a cell vector
                m = len(cellDFB[i])
                for j in xrange(0, m):
                    dSum = dSum + sum((abs(cellDFB[i][j]) - dAbsMean)**2)
            else:
                dSum = dSum + sum((abs(cellDFB{i}) - dAbsMean)**2)
        dStd = sqrt(dSum / (nCount - 1))

    # Modified on 04/01/04
    # Compute the scale values
    vScales[0] = max(dAbsMean - dRatio * dStd, dAbsMin)
    vScales[1] = min(dAbsMean + dRatio * dStd, dAbsMax)
