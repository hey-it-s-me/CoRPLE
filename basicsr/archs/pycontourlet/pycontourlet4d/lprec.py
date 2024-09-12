from numpy import *
from dup import *
from sefilter2 import *

def lprec(c, d, h, g):
    """ LPDEC   Laplacian Pyramid Reconstruction
    
    x = lprec(c, d, h, g)
    
    Input:
    c:      coarse image at half size
    d:      detail image at full size
    h, g:   two lowpass filters for the Laplacian pyramid
    
    Output:
    x:      reconstructed image
    
    Note:     This uses a new reconstruction method by Do and Vetterli,
    Framming pyramids, IEEE Trans. on Sig Proc., Sep. 2003.
    
    See also:	LPDEC, PDFBREC"""

    # First, filter and downsample the detail image
    xhi = sefilter2(d, h, h, 'per', None)
    xhi = xhi[::2, ::2]

    # Subtract from the coarse image, and then upsample and filter
    xlo = c - xhi
    xlo = dup(xlo, array([2, 2]), None)

    # Even size filter needs to be adjusted to obtain
    # perfect reconstruction with zero shift
    adjust = mod(len(g) + 1, 2)

    xlo = sefilter2(xlo, g, g, 'per', adjust * array([1, 1]))

    # Final combination
    x = xlo + d
    
    return x
