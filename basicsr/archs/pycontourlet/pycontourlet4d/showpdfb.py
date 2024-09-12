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
from matplotlib import *


def showpdfb(y, scaleMode, displayMode, lowratio,
             highratio, coefMode, subbandgap):
    """ SHOWPDFB   Show PDFB coefficients.

    showpdfb(y, [scaleMode, displayMode, lowratio,
    highratio, coefMode, subbandgap])

    Input:
    y:	    a cell vector of length n+1, one for each layer of
    subband images from DFB, y{1} is the lowpass image

    scaleMode:
    scale mode (a string or number):
    If it is a number, it denotes the number of most significant
    coefficients to be displayed.  Its default value is 'auto2'.
    'auto1' ---   All the layers use one scale. It reflects the real
    values of the coefficients.
    However, the visibility will be very poor.
    'auto2' ---   Lowpass uses the first scale. All the highpass use
    the second scale.
    'auto3' ---   Lowpass uses the first scale.
    All the wavelet highpass use the second scale.
    All the contourlet highpass use the third scale.

    displayMode:
    display mode (a string):
    'vb' -----  display the layers vertically in Matlab environment.
    It uses the background color for the marginal
    image.
    'vw' -----  display the layers vertically for print.
    It uses the white color for the marginal
    image.
    'hb' -----  display the layers horizontally in Matlab environment.
    It uses the background color for the marginal
    image.
    'hw' -----  display the layers horizontally for print.
    It uses the white color for the marginal
    image.

    lowratio:
    display ratio for the lowpass filter (default value is 2).
    It ranges from 1.2 to 4.0.
    highratio:
    display ratio for the highpass filter (default value is 6).
    It ranges from 1.5 to 10.

    coefMode:
    coefficients mode (a string):
    'real' ----  Highpass filters use the real coefficients.
    'abs' ------ Highpass filters use the absolute coefficients.
    It is the default value

    subbandgap:
    gap (in pixels) between subbands. It ranges from 1 to 4.

    Output:
    displayIm:  matrix for the display image.

    See also:     PDFBDEC, DFBIMAGE, COMPUTESCALE

    History:
    09/17/2003  Creation.
    09/18/2003  Add two display mode, denoted by 'displayMode'.
    Add two coefficients mode, denoted by 'coeffMode'.
    10/03/2003  Add the option for the lowpass wavelet decomposition.
    10/04/2003  Add a function computescale in computescales.m.
    This function will call it.
    Add two scal modes, denoted by 'scaleMode'.
    It can also display the most significant coefficients.
    10/05/2003  Add 'axis image' to control resizing.
    Use the two-fold searching method to find the
    background color index.
    04/01/2004  Fixed a bug.
    04/08/2004  Add new display modes and display the layer images
    horizontally. """

    if y is None:
        print "Read showpdfb doc"
        return
    # Scale mode
    if scaleMode is None:
        scaleMode = 'auto2'
    elif elseif isnumeric(scaleMode):
        # Denote the number of significant coefficients to be displayed
        if scaleMode < 2:
            print 'Warning! The number of significant coefficients must be
            positive!'
            scaleMode = 50
    elif ~strcmp(scaleMode, 'auto1') and ~strcmp(scaleMode, 'auto2')
    and ~strcmp(scaleMode, 'auto3'):
        print 'Warning! There are only two scaleMode mode: \n
        auto1, auto2, auto3! Its defualt value is auto2!'
        scaleMode = 'auto2'
    # Display ratio for the lowpass band
    if lowratio is None:
        lowratio = 2
    elif lowratio < 1:
        print print 'Warning! lowratio must be larger than 1!\n
        Its defualt value is 2!'
        lowratio = 2

    # Display ratio for the hiphpass band
    if highratio is None:
        highratio = 6
    elif highratio < 1:
        print 'Warning! highratio must be larger than 1!\n
        Its defualt value is 6!'
        highratio = 6

    # Gap between subbands
    if subbandgap is None:
        subbandgap = 1
    elif subbandgap < 1:
        print 'Warning! subbandgap must be no less than 1
        \nIts defualt value is 1!'
        subbandgap = 1

    # Display mode
    if displayMode is None:
        displayMode = 'hw'
    elif ~strcmp(displayMode, 'vb') and ~strcmp(displayMode, 'vw')
    and ~strcmp(displayMode, 'hb') and ~strcmp(displayMode, 'hw'):
        print 'Warning! There are only four display mode: "vb",
        "vw", "hb", "hw"!\nIts defualt value is "vb"!'
        displayMode = 'vw'

    # Coefficient mode
    if coefMode is None:
        coefMode = 'abs'
    elif ~strcmp(coefMode, 'real') & ~strcmp(coefMode, 'abs'):
        print 'Warning! There are only two coefficients mode: real, abs!
        \nIts defualt value is "abs"!'
        coefMode = 'abs'

    # Parameters for display
    layergap = 1  # Gap between layers

    # Input structure analysis.
    nLayers = len(y)  # number of PDFB layers
    # Compute the number of wavelets layers.
    # We assume that the wavelets layers are first several consecutive layers.
    # The number of the subbands of each layer is 3.
    fWaveletsLayer = 1
    nWaveletsLayers = 0  # Number of wavelets layers.
    nInxContourletLayer = 0  # The index of the first contourlet layer.
    i = 2

    while fWaveletsLayer > 0 and i <= nLayers:
        if len(y[i]) == 3:
            nWaveletsLayers = nWaveletsLayers + 1
        else:
            fWaveletsLayer = 0
        i = i + 1

    nInxContourletLayer = 2 + nWaveletsLayers

    # Initialization
    # Since we will merge the wavelets layers together,
    # we shall decrease the number of display layers.

    nDisplayLayers = nLayers - nWaveletsLayers
    cellLayers = [list()] * nDisplayLayers  # Cell for multiple display layers
    vScalesTemp = zeros((1, 2))   # Temporary scale vector.
    vScales = zeros((nLayers, 2))  # Scale vectors for each layer
    nAdjustHighpass = 2  # Adjustment ratio for the highpass layers.

    if ~isnumeric(scaleMode):
        if scaleMode == 'auto1':  # Compute the scales for each layer
            vScalesTemp = computescale(y, lowratio, 1, nLayers, coefMode)
            for i in xrange(0, nLayers):
                vScales[i, :] = vScalesTemp
        elif scaleMode == 'auto2':
            vScales[0, :] = computescale(y, lowratio, 1, 1, coefMode)
            vScalesTemp = computescale(y, highratio, 2, nLayers, coefMode)
            # Make a slight adjustment.
            # Compared to the lowpass, the highpass shall be insignificant.
            # To make the display more realistic,
            # use a little trick to make the upper bound a little bigger.
            # vScalesTemp[1] = nAdjustHighpass
            # * ( vScalesTemp[1] - vScalesTemp[0] ) + vScalesTemp[0] ;
            for i in xrange(1: nLayers):
                vScales[i, :] = vScalesTemp
        elif scaleMode == 'auto3':
            vScales[1, :] = computescale(y, lowratio, 1, 1, coefMode)
            vScalesTemp = computescale(y, highratio, 2,
                                        1 + nWaveletsLayers, coefMode)
            # Make a slight adjustment.
            # Compared to the lowpass, the highpass shall be insignificant.
            # To make the display more realistic,
            # use a little trick to make the upper bound a little bigger.
            vScalesTemp[1] = nAdjustHighpass *
            (vScalesTemp[1] - vScalesTemp[0]) + vScalesTemp[0]
            for i in xrange(1, nWaveletsLayers + 1):
                vScales[i, :] = vScalesTemp
            vScalesTemp = computescale(y,
                                        highratio, nInxContourletLayer,
                                        nLayers, coefMode)
            # Make a slight adjustment. Compared to the lowpass,
            # the highpass shall be insignificant.
            # To make the display more realistic,
            # use a little trick to make the upper bound a little bigger.
            vScalesTemp[1] = nAdjustHighpass *
            (vScalesTemp[1] - vScalesTemp[0]) + vScalesTemp[0]
            for i in xrange(nInxContourletLayer, nLayers):
                vScales[i, :] = vScalesTemp
        else:  # Default value: 'auto2'.
            vScales[1, :] = computescale(y, lowratio, 1, 1, coefMode)
            vScalesTemp = computescale(y, highratio, 2, nLayers, coefMode)
            for i in xrange(1, nLayers):
            vScales[i, :] = vScalesTemp
    # Verify that they are reasonable
    for i in xrange(0, nLayers):
        if vScales[i, 1] < vScales[i, 0] + 1.0e-9
            print 'Error in showpdfb.m! The scale vectors are wrong!'
    # display ( vScales ) ;
    else:  # Compute the threshold for the display of coefficients
        # Convert the output into the vector format
        [vCoeff, s] = pdfb2vec(y)

    # Sort the coefficient in the order of energy.
    vSort = sort(abs(vCoeff))
    # clear vCoeff
    vSort = fliplr(vSort)

    # Find the threshold value based on number of keeping coeffs
    dThresh = vSort(scaleMode)
    # clear vSort;

    # Prepare for the display
    colormap(gray)
    cmap = get(gcf, 'Colormap')
    cColorInx = size(cmap, 1)

    # Find background color index:
    if strcmp(displayMode, 'vb') or strcmp(displayMode, 'hb'):
    # Get the background color (gray value)
    dBgColor = get(gcf, 'Color')

    # Search the color index by 2-fold searching method.
    # This method is only useful for the gray color!
    nSmall = 1
    nBig = cColorInx
    while nBig > nSmall + 1:
        nBgColor = floor((nSmall + nBig) / 2.0)
        if dBgColor(1) < cmap(nBgColor, 1):
            nBig = nBgColo
        else:
            nSmall = nBgColor

    if abs(dBgColor(1) - cmap(nBig, 1)) >
    abs(dBgColor(1) - cmap(nSmall, 1)):
        nBgColor = nSmall
    else:
        nBgColor = nBig

    # Merge all layers to corresponding display layers.
    # Prepare the cellLayers, including the boundary.
    # Need to polish with real boudary later!
    # Now we add the boundary, but erase the images!!
    # First handle the lowpass filter
    # White line around subbands

    # 1. One wavelets layers.
    gridI = cColorInx - 1;
    cell4Wavelets = [list()] * 4  # Store 4 wavelets subbands.
    if isnumeric(scaleMode):  # Keep the significant efficients
        waveletsIm = cColorInx * double(abs(y[0]) >= dThresh)
    else:
        dRatio = (cColorInx - 1) / (vScales[0, 1] - vScales[0, 0])
    if strcmp(coefMode, 'real'):
        waveletsIm = double(1 + (y[0] - vScales[0, 0] * dRatio)
    else:
        waveletsIm=double(1 + (abs(y[0]) - vScales[0, 0]) * dRatio)

    # Merge other wavelets layers
    if nWaveletsLayers > 0:
        for i in xrange(1, nWaveletsLayers + 1):
            cell4Wavelets[0]=waveletsIm
            # Compute with the scale ratio.
            if ~isnumeric(scaleMode):
                dRatio=(cColorInx - 1) / (vScales[i, 1] - vScales[i, 0])

                m=len(y[i]);
            if m |= 3:
                print 'Error in showpdfb.m!
                Incorect number of wavelets subbands!'
                for k in xrange(0, m):
                    if isnumeric(scaleMode):  # Keep the significant efficients
                        cell4Wavelets[k + 1]=cColorInx
                        * double(abs(y[i][k]) >= dThresh)
                    else:
                        if strcmp(coefMode, 'real'):
                            cell4Wavelets[k + 1]=double(1 + (y[i][k] - vScales[i, 0]) * dRatio)
                        else:
                            cell4Wavelets{k + 1}=double(1 + (abs(y[i][k]) - vScales[i, 0]) * dRatio)
            waveletsIm=dfbimage(cell4Wavelets, subbandgap, gridI)

    cellLayers[0]=waveletsIm

    # Compute the inital size of the dispaly image.
    if strcmp(displayMode, 'vb') or strcmp(displayMode, 'vw'):
        # Display vertically
        nHeight=size(cellLayers[0], 1)
    else:
        # Display horizontally
        nWidth=size(cellLayers[0], 2)

    # 2. All the contourlet layers
    for i in xrange(nInxContourletLayer, nLayers):
        # Compute with the scale ratio.
        if ~isnumeric(scaleMode):
            dRatio=(cColorInx - 1) / (vScales[i, 1] - vScales[i, 0])

    m=len(y[i])
    z=[[None]] * m
    for k in xrange(o, m):
        if isnumeric(scaleMode):  # Keep the significant efficients
            z[k]=cColorInx * double(abs(y[i][k]) >= dThresh)
        else:
            if strcmp(coefMode, 'real'):
                z[k]=double(1 + (y[i][k] - vScales[i, 0]) * dRatio)
            else:
                z[k]=double(1 + (abs(y[i][k]) - vScales[i, 0]) * dRatio)

    cellLayers{i - nWaveletsLayers}=dfbimage(z, subbandgap, gridI)

    # Update the size
    if strcmp(displayMode, 'vb') or strcmp(displayMode, 'vw'):
        # Display vertically
        nHeight=nHeight + size(cellLayers[i - nWaveletsLayers], 1)
    else:
        # Display horizontally
        nWidth=nWidth + size(cellLayers[i - nWaveletsLayers], 2)

    # Merge all layers and add gaps between layers
    if strcmp(displayMode, 'vb') or strcmp(displayMode, 'vw'):
        # Display vertically
        nWidth=size(cellLayers[nDisplayLayers], 2)
        nHeight=nHeight + layergap * (nDisplayLayers - 1)
    else:
        # Display horizontally
        nHeight=size(cellLayers[nDisplayLayers], 1)
        nWidth=nWidth + layergap * (nDisplayLayers - 1)

    # Set the background for the output image
    if strcmp(displayMode, 'vb') or strcmp(displayMode, 'hb'):
        displayIm=nBgColor * ones(nHeight, nWidth)
    else:
        displayIm=(cColorInx - 1) * ones((nHeight, nWidth))

    nPos=0  # output image pointer

    if strcmp(displayMode, 'vb') or strcmp(displayMode, 'vw'):
        # Display vertically
        for i in xrange(0, nDisplayLayers):
            h, w=cellLayers[i].shape
            displayIm(nPos + 1: nPos + h, 0: w)=cellLayers[i]
        if i < nDisplayLayers:
            # Move the position pointer and add gaps between layers
            nPos=nPos + h + layergap

    else:
    # Display horizontally

    for i in xrange(0, nDisplayLayers):
        h, w=cellLayers[i].shape
        displayIm(0: h, nPos + 1: nPos + w)=cellLayers[i]

        if i < nDisplayLayers:
            # Move the position pointer and add gaps between layers
            nPos=nPos + w + layergap

    hh=image(displayIm)
    % title('decompostion image')
    axis image off
