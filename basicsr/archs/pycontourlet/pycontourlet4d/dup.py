from numpy import *

def dup(x, step, phase):
    """ DUP   Diagonal Upsampling
    
    y = dup(x, step, [phase])
    
    Input:
    x:	input image
    step:	upsampling factors for each dimension which should be a
    2-vector
    phase:	[optional] to specify the phase of the input image which
    should be less than step, (default is [0, 0])
    If phase == 'minimum', a minimum size of upsampled image
    is returned
    
    Output:
    y:	diagonal upsampled image
    
    See also:	DDOWN"""

    if phase is None:
        phase = array([0, 0])

    sx = array(x.shape)
    
    if phase[0] == 'm' or phase[0] == 'M':
        y = zeros((sx - 1) * step + 1)
        y[0::step[0], 0::step[0]] = x.copy()
    else:
        y = zeros(sx * step)
        y[phase[0]::step[0], phase[1]::step[1]] = x.copy()
    return y
