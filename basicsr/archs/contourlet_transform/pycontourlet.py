from numpy import *
from scipy import signal
from .dfilters import dfilters
from .modulate2 import modulate2
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})
from .resamp4c import resamp4c
import torch
from torch.nn import functional as F

def batch_multi_channel_pdfbdec(x, pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0, 3, 3, 3], device=torch.device("cpu")):
    """Multi-channel pyramidal directional filter bank decomposition
     for a batch of images.

        Parameters
        ----------
        x : 4D Tensor
            Tensor in the following dimension:
            (batch_size, channel_size, image_width, image_height)
        pfilt: str, default="maxflat"
            Filter name for the pyramidal decomposition step
        dfilt: str, default="dmaxflat7"
            Filter name for the directional decomposition step
        n_levs : list of int, default=[0, 3, 3, 3]
             The numbers of DFB (Directional Filter Bank) decomposition levels at
             each pyramidal level from coarse to fine-scale.

             In each pyramidal level, there is one number of DFB decomposition
             levels is denoted as `l`, resulting in `2^l` wedge-shaped subbands
             in the frequency domain.  0 denoted a 2-D wavelet decomposition,
             resulting a coarse approximation and three bandpass directional
             subbands in scale 4.

             For example:
                 mode = 'resize'
                 n_channels = 3
                 n_levs = [0, 3, 3, 3]
                 num_subbands = [(1+3)*3, (2^3)*3, (2^3)*3, (2^3)*3]
                              = [12, 24, 24, 24]
        device : torch.device, default=torch.device("cpu")
            The device to store and compute Torch tensor.
            Either "cpu" or "cuda".

        Returns
        -------
        coefs : dictionary of 4D numpy array
            Dictionary containing the coefficients obtained from the
            decomposition, with each level as the key. The index key
            starts is the numpy array shape.
            The dictionary is in the following format:
                ```coefs[level]```
            Here's an example with a batch of 2 images with 3 channels,
            and the following settings:
                >>> x.shape
                torch.Size([2, 3, 224, 224])
                >>> y = batch_multi_channel_pdfbdec(x, n_levs=[0, 3, 3, 3])
            This will yield:
                >>> y[(2, 3, 14, 14)].shape
                (2, 3, 14, 14)
                >>> y[(2, 3, 14, 28)].shape
                (2, 3, 14, 28)
                >>> y[(2, 3, 28, 14)].shape
                (2, 3, 28, 14)
                >>> y[(2, 3, 56, 28)].shape
                (2, 3, 56, 28)
                >>> y[(2, 3, 56, 112)].shape
                (2, 3, 56, 112)
                >>> y[(2, 3, 112, 56)].shape
                (2, 3, 112, 56)
    """
    if len(nlevs) == 0:
        y = [x]
    else:
        # Get the pyramidal filters from the filter name
        h, g = pfilters(pfilt)
        if nlevs[-1] != 0:
            # Laplacian decomposition
            xlo, xhi = lpdec(x, h, g, device=device)
            # DFB on the bandpass image
            if dfilt == 'pkva6' or dfilt == 'pkva8' or dfilt == 'pkva12' or dfilt == 'pkva':
                # Use the ladder structure (whihc is much more efficient)
                xhi_dir = dfbdec_l(xhi, dfilt, nlevs[-1])
            else:
                # General case
                xhi_dir = dfbdec(xhi, dfilt, nlevs[-1])

        else:
            # Special case: nlevs(end) == 0
            # Perform one-level 2-D critically sampled wavelet filter bank
            xlo, xLH, xHL, xHH = wfb2dec(x, h, g)
            xhi_dir = [xLH]
            xhi_dir.append(xHL)
            xhi_dir.append(xHH)

        # Recursive call on the low band
        ylo = batch_multi_channel_pdfbdec(xlo, pfilt, dfilt, nlevs[0:-1])

        # Add bandpass directional subbands to the final output
        y = ylo[:]
        y.append(xhi_dir)
        
    return y

def dfbdec(x, fname, n, device=torch.device("cpu")):
    """ DFBDEC   Directional Filterbank Decomposition

    y = dfbdec(x, fname, n)

    Input:
    x:      input image
    fname:  filter name to be called by DFILTERS
    n:      number of decomposition tree levels

    Output:
    y:      subband images in a cell vector of length 2^n

    Note:
    This is the general version that works with any FIR filters

    See also: DFBREC, FBDEC, DFILTERS"""
    if (n != round(n)) or (n < 0):
        print('Number of decomposition levels must be a non-negative integer')

    if n == 0:
        # No decomposition, simply copy input to output
        y = [None]
        y[0] = x.copy()
        return y

    # Get the diamond-shaped filters
    h0, h1 = dfilters(fname, 'd')

    # Fan filters for the first two levels
    # k0: filters the first dimension (row)
    # k1: filters the second dimension (column)
    k0 = modulate2(h0, 'c', None)
    k1 = modulate2(h1, 'c', None)
    # Tree-structured filter banks
    if n == 1:
        # Simplest case, one level
        y = [[None]] * 2
        y[0], y[1] = fbdec(x, k0, k1, 'q', '1r', 'per')
    else:
        # For the cases that n >= 2
        # First level
        x0, x1 = fbdec(x, k0, k1, 'q', '1r', 'per')
        # Second level
        y = [[None]] * 4
        y[0], y[1] = fbdec(x0, k0, k1, 'q', '2c', 'qper_col')
        y[2], y[3] = fbdec(x1, k0, k1, 'q', '2c', 'qper_col')
        # Fan filters from diamond filters
        f0, f1 = ffilters(h0, h1)
        # Now expand the rest of the tree
        for l in range(3, n + 1):
            # Allocate space for the new subband outputs
            y_old = y[:]
            y = [[None]] * 2**l
            # The first half channels use R1 and R2
            for k in range(0, 2**(l - 2)):
                i = mod(k, 2)
                y[2 * k], y[2 * k + 1] = fbdec(y_old[k],
                                               f0[i], f1[i], 'pq', i, 'per')
            # The second half channels use R3 and R4
            for k in range(2**(l - 2), 2**(l - 1)):
                i = mod(k, 2) + 2
                y[2 * k], y[2 * k + 1] = fbdec(y_old[k],
                                               f0[i], f1[i], 'pq', i, 'per')
    # Back sampling (so that the overal sampling is separable)
    # to enhance visualization
    y = backsamp(y)
    # Flip the order of the second half channels
    y[2**(n - 1)::] = y[::-1][:2**(n - 1)]

    return y

def pfilters(fname, device=torch.device("cpu")):
    """ PFILTERS Generate filters for the laplacian pyramid

    Input:
    fname : Name of the filter, including the famous '9-7' filters.

    Output:
    h, g: 1D filters (lowpass for analysis and synthesis, respectively)
    for separable pyramid"""

    if fname == "9/7" or fname == "9-7":
        h = array([.037828455506995, -.023849465019380, -.11062440441842,
                   .37740285561265])
        h = hstack((h, .85269867900940, h[::-1]))

        g = array([-.064538882628938, -.040689417609558, .41809227322221])
        g = hstack((g, .78848561640566, g[::-1]))

        return h, g
    elif fname == "maxflat":
        M1 = 1 / sqrt(2)
        M2 = M1
        k1 = 1 - sqrt(2)
        k2 = M1
        k3 = k1
        h = array([.25 * k2 * k3, .5 * k2, 1 + .5 * k2 * k3]) * M1
        h = hstack((h, h[len(h) - 2::-1]))

        g = array([-.125 * k1 * k2 * k3, 0.25 * k1 * k2, -0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3,
                   1 + .5 * k1 * k2]) * M2
        g = hstack((g, g[len(g) - 2::-1]))
        # Normalize
        h = h * sqrt(2)
        g = g * sqrt(2)
        return h, g
    elif fname == "5/3" or fname == "5-3":
        h = [-1, 2, 6, 2, -1] / (4 * sqrt(2))
        g = [1, 2, 1] / (2 * sqrt(2))
        return h, g
    elif fname == "burt" or fname == "Burt":
        h = array([0.6, 0.25, -0.05])
        h = sqrt(2) * hstack((h[len(h):0:-1], h))

        g = array([17.0 / 28, 73.0 / 280, -3.0 / 56, -3.0 / 280])
        g = sqrt(2) * hstack((g[len(g):0:-1], g))
        return h, g
    elif fname == "pkva":
        # filters from the ladder structure
        # Allpass filter for the ladder structure network
        beta = ldfilter(fname)

        lf = len(beta)
        n = float(lf) / 2

        if n != floor(n):
            print("The input allpass filter must be even length")

        # beta(z^2)
        beta2 = zeros(2 * lf - 1)
        beta2[::2] = beta

        # H(z)
        h = beta2.copy()
        h[2 * n - 1] = h[2 * n - 1] + 1
        h = h / 2

        # G(z)
        g = -convolve(beta2, h)
        g[4 * n - 2] = g[4 * n - 2] + 1
        g[1:-1:2] = -g[1:-1:2]

        # Normalize
        h = h * sqrt(2)
        g = g * sqrt(2)
        return h, g

def lpdec(x, h, g, device=torch.device("cpu")):
    """ LPDEC   Laplacian Pyramid Decomposition

    [c, d] = lpdec(x, h, g)

    Input:
    x:      input image
    h, g:   two lowpass filters for the Laplacian pyramid

    Output:
    c:      coarse image at half size
    d:      detail image at full size

    See also:   LPREC, PDFBDEC"""

    # Lowpass filter and downsample
    xlo = sefilter2(x, h, h, 'per', None, device=device)
    c = xlo[:, :, ::2, ::2].cpu()
    
    # Compute the residual (bandpass) image by upsample, filter, and subtract
    # Even size filter needs to be adjusted to obtain perfect reconstruction
    adjust = mod(len(g) + 1, 2)

    xlo = zeros(x.shape)
    xlo[:, :, ::2, ::2] = c
    tmp = sefilter2(xlo, g, g, 'per', adjust * array([1, 1]), device=device)
    d = x.to(device) - tmp.to(device)

    return c, d

def sefilter2(x, f1, f2, extmod, shift, device=torch.device("cpu")):
    """SEFILTER2   2D separable filtering with extension handling
    y = sefilter2(x, f1, f2, [extmod], [shift])

    Input:
    x:      input image
    f1, f2: 1-D filters in each dimension that make up a 2D seperable filter
    extmod: [optional] extension mode (default is 'per')
    shift:  [optional] specify the window over which the
    convolution occurs. By default shift = [0; 0].

    Output:
    y:      filtered image of the same size as the input image:
    Y(z1,z2) = X(z1,z2)*F1(z1)*F2(z2)*z1^shift(1)*z2^shift(2)

    Note:
    The origin of the filter f is assumed to be floor(size(f)/2) + 1.
    Amount of shift should be no more than floor((size(f)-1)/2).
   The output image has the same size with the input image.

   See also: EXTEND2, EFILTER2"""

    if extmod is None:
        extmod = 'per'

    if shift is None:
        shift = array([[0], [0]])

    # Make sure filter in a row vector
    f1 = f1[:, np.newaxis].reshape(len(f1),)
    f2 = f2[:, np.newaxis].reshape(len(f2),)
    
    # Periodized extension
    lf1 = (len(f1) - 1) / 2.0
    lf2 = (len(f1) - 1) / 2.0
    y = extend2(x, floor(lf1) + shift[0], ceil(lf1) - shift[0],
                floor(lf2) + shift[1], ceil(lf2) - shift[1], extmod)

    # Seperable filter
    if not torch.is_tensor(y):
        inputs = torch.from_numpy(y.astype(np.float32)).to(device)
    else:
        inputs = y.type(torch.FloatTensor).to(device)
    filters = torch.from_numpy((f1[:, np.newaxis] * f2).astype(np.float32)).to(device)
    filters = filters[None, None, :, :]
    filters = torch.repeat_interleave(filters, y.shape[1], axis=0)
    
    y = F.conv2d(inputs, filters, groups=inputs.size(1))

    return y

def extend2(x, ru, rd, cl, cr, extmod, device=torch.device("cpu")):
    """ EXTEND2   2D extension
    y = extend2(x, ru, rd, cl, cr, extmod)

    Input:
    x:  input image
    ru, rd: amount of extension, up and down, for rows
    cl, cr: amount of extension, left and rigth, for column
    extmod: extension mode.  The valid modes are:
    'per':      periodized extension (both direction)
    'qper_row': quincunx periodized extension in row
    'qper_col': quincunx periodized extension in column

    Output:
    y:  extended image

    Note:
    Extension modes 'qper_row' and 'qper_col' are used multilevel
    quincunx filter banks, assuming the original image is periodic in
    both directions.  For example:
    [y0, y1] = fbdec(x, h0, h1, 'q', '1r', 'per');
    [y00, y01] = fbdec(y0, h0, h1, 'q', '2c', 'qper_col');
    [y10, y11] = fbdec(y1, h0, h1, 'q', '2c', 'qper_col');

    See also:   FBDEC"""

    _, _, rx, cx = array(x.shape)

    if extmod == 'per':

        I = getPerIndices(rx, ru, rd)
        y = x[:, :, I, :]

        I = getPerIndices(cx, cl, cr)
        y = y[:, :, :, I]

        return y
    elif extmod == 'qper_row':
        rx2 = round(rx / 2.0)
        y = np.c_[np.r_[x[:, :, rx2:rx, cx - cl:cx], x[:, :, 0:rx2, cx - cl:cx]],
               x, np.r_[x[:, :, rx2:rx, 0:cr], x[:, :, 0:rx2, 0:cr]]]
        I = getPerIndices(rx, ru, rd)
        y = y[:, :, I, :]
        return y
    elif extmod == 'qper_col':
        cx2 = int(round(cx / 2.0))
        y = np.concatenate([
            np.c_[x[:, :, rx - ru:rx, cx2:cx], x[:, :, rx - ru:rx, 0:cx2]],
            x,
            np.c_[x[:, :, 0:rd, cx2:cx], x[:, :, 0:rd, 0:cx2]]
        ], axis=2)

        I = getPerIndices(cx, cl, cr)
        y = y[:, :, :, I]
        return y
    else:
        print("Invalid input for EXTMOD")

def getPerIndices(lx, lb, le, device=torch.device("cpu")):
    I = hstack((arange(lx - lb, lx), arange(0, lx), arange(0, le)))
    if (lx < lb) or (lx < le):
        I = mod(I, lx)
        I[I == 0] = lx
    return I.astype(int)

def fbdec(x, h0, h1, type1, type2, extmod, device=torch.device("cpu")):
    """ FBDEC   Two-channel 2D Filterbank Decomposition

    [y0, y1] = fbdec(x, h0, h1, type1, type2, [extmod])

    Input:
    x:  input image
    h0, h1: two decomposition 2D filters
    type1:  'q', 'p' or 'pq' for selecting quincunx or parallelogram
    downsampling matrix
    type2:  second parameter for selecting the filterbank type
    If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
    If type1 == 'p' then type2 is one of {0, 1, 2, 3}
    Those are specified in QDOWN and PDOWN
    If type1 == 'pq' then same as 'p' except that
    the paralellogram matrix is replaced by a combination
    of a  resampling and a quincunx matrices
    extmod: [optional] extension mode (default is 'per')

    Output:
    y0, y1: two result subband images

    Note:       This is the general implementation of 2D two-channel
    filterbank

    See also:   FBDEC_SP """

    # For parallegoram filterbank using quincunx downsampling, resampling is
    # applied before filtering
    if type1 == 'pq':
        x = resamp(x, type2, None, None, device=device)

    # Stagger sampling if filter is odd-size (in both dimensions)
    if all(mod(h1.shape, 2)):
        shift = array([[-1], [0]])

        # Account for the resampling matrix in the parallegoram case
        if type1 == 'p':
            R = [[None]] * 4
            R[0] = array([[1, 1], [0, 1]])
            R[1] = array([[1, -1], [0, 1]])
            R[2] = array([[1, 0], [1, 1]])
            R[3] = array([[1, 0], [-1, 1]])
            shift = R[type2] * shift
    else:
        shift = array([[0], [0]])
    # Extend, filter and keep the original size
    y0 = efilter2(x, h0, extmod, None)
    y1 = efilter2(x, h1, extmod, shift)
    # Downsampling
    if type1 == 'q':
        # Quincunx downsampling
        y0 = qdown(y0, type2, None, None, device=device)
        y1 = qdown(y1, type2, None, None, device=device)
    elif type1 == 'p':
        # Parallelogram downsampling
        y0 = pdown(y0, type2, None, None)
        y1 = pdown(y1, type2, None, None)
    elif type1 == 'pq':
        # Quincux downsampling using the equipvalent type
        pqtype = ['1r', '2r', '2c', '1c']
        y0 = qdown(y0, pqtype[type2], None, None)
        y1 = qdown(y1, pqtype[type2], None, None)
    else:
        print("Invalid input type1")

    return y0, y1

def efilter2(x, f, extmod, shift, device=torch.device("cpu")):
    """EFILTER2   2D Filtering with edge handling (via extension)

    y = efilter2(x, f, [extmod], [shift])

    Input:
    x:  input image
    f:  2D filter
    extmod: [optional] extension mode (default is 'per')
    shift:  [optional] specify the window over which the
    convolution occurs. By default shift = [0; 0].

    Output:
    y:  filtered image that has:
    Y(z1,z2) = X(z1,z2)*F(z1,z2)*z1^shift(1)*z2^shift(2)

    Note:
    The origin of filter f is assumed to be floor(size(f)/2) + 1.
    Amount of shift should be no more than floor((size(f)-1)/2).
    The output image has the same size with the input image.

    See also:   EXTEND2, SEFILTER2"""

    if extmod is None:
        extmod = 'per'
    if shift is None:
        shift = array([[0], [0]])

    # Periodized extension
    if f.ndim < 2:
        sf = (r_[1, array(f.shape)] - 1) / 2.0
    else:
        sf = (array(f.shape) - 1) / 2.0

    ru = int(floor(sf[0]) + shift[0][0])
    rd = int(ceil(sf[0]) - shift[0][0])
    cl = int(floor(sf[1]) + shift[1][0])
    cr = int(ceil(sf[1]) - shift[1][0])
    xext = extend2(x, ru, rd, cl, cr, extmod)

    # Convolution and keep the central part that has the size as the input
    if f.ndim < 2:
        inputs = xext.t.to(device)
        filters = torch.from_numpy((f1[:, np.newaxis]).astype(np.float32)).to(device)
        filters = filters[None, None, :, :]
        filters = torch.repeat_interleave(filters, y.shape[1], axis=0)

        y = F.conv2d(inputs, filters, groups=inputs.size(1)).t

    else:
        if not torch.is_tensor(xext):
            inputs = torch.from_numpy(xext.astype(np.float32)).to(device)
        else:
            inputs = xext.to(device)
        filters = torch.from_numpy(f.astype(np.float32))
        filters = filters[None, None, :, :]
        filters = torch.repeat_interleave(filters, x.shape[1], axis=0)

        y = F.conv2d(inputs, filters, groups=inputs.size(1))

    return y

# def efilter2(x, f, extmod='per', shift=[0, 0], device=torch.device("cpu")):
#     # 假设x已经是一个PyTorch张量
#     if not torch.is_tensor(x):
#         inputs = torch.from_numpy(x.astype(np.float32)).to(device)
#     else:
#         inputs = x.to(device)
#
#     if f.ndim == 2:
#         f = np.expand_dims(f, axis=0)  # 为滤波器增加一个“通道”维度
#     filters = torch.from_numpy(f.astype(np.float32)).to(device)
#     filters = filters.unsqueeze(0)  # 在滤波器前增加一个批次维度
#
#     # 如果是多通道图像，重复滤波器以匹配输入通道数
#     if inputs.shape[1] > 1:
#         filters = filters.repeat(inputs.shape[1], 1, 1, 1)
#         groups = inputs.shape[1]  # 每个通道独立卷积
#     else:
#         groups = 1
#
#     # 应用卷积
#     y = F.conv2d(inputs, filters, groups=groups, padding=(shift[0], shift[1]))
#
#     return y


def qdown(x, type, extmod, phase, device=torch.device("cpu")):
    """% QDOWN   Quincunx Downsampling
    %
    %   y = qdown(x, [type], [extmod], [phase])
    %
    % Input:
    %   x:  input image
    %   type:   [optional] one of {'1r', '1c', '2r', '2c'} (default is '1r')
    %       '1' or '2' for selecting the quincunx matrices:
    %           Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
    %       'r' or 'c' for suppresing row or column
    %   phase:  [optional] 0 or 1 for keeping the zero- or one-polyphase
    %       component, (default is 0)
    %
    % Output:
    %   y:  qunincunx downsampled image
    %
    % See also: QPDEC"""

    if type is None:
        type = '1r'

    if phase is None:
        phase = 0

    if type == '1r':
        z = resamp(x, 1, None, None, device=device)
        if phase == 0:
            y = resamp(z[:, :, ::2, :], 2, None, None, device=device)
        else:
            y = resamp(hstack((z[:, :, 1::2, 1:], z[:, :, 1::2, 0:1])), 2, None, None, device=device)

    elif type == '1c':
        z = resamp(x, 2, None, None, device=device)
        if phase == 0:
            y = resamp(z[:, :, :, ::2], 1, None, None, device=device)
        else:
            y = resamp(z[:, :, :, 1::2], 1, None, None, device=device)
    elif type == '2r':
        z = resamp(x, 0, None, None, device=device)
        if phase == 0:
            y = resamp(z[:, :, ::2, :], 3, None, None, device=device)
        else:
            y = resamp(z[:, :, 1::2, :], 3, None, None, device=device)
    elif type == '2c':
        z = resamp(x, 3, None, None, device=device)
        if phase == 0:
            y = resamp(z[:, :, :, ::2], 0, None, None, device=device)
        else:
            y = resamp(hstack((z[:, :, 1:, 1::2].conj().transpose(0, 1, 3, 2),
                               z[:, :, 0:1, 1::2].conj().transpose(0, 1, 3, 2))).conj().transpose(0, 1, 3, 2), 
                       0, None, None, device=device)
    else:
        print("Invalid argument type")
    return y

def resamp(x, type_, shift, extmod, device=torch.device("cpu")):
    """ RESAMP   Resampling in 2D filterbank

        y = resamp(x, type, [shift, extmod])

        Input:
        x:  input image
        type: one of {0,1,2,3} (see note)

        shift:  [optional] amount of shift (default is 1)
        extmod: [optional] extension mode (default is 'per').
        Other options are:

        Output:
        y:  resampled image.

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

    # Convert to np.float32
    if torch.is_tensor(x):
        x = x.numpy()
    else:
        x = x.astype(np.float32)
    
    if shift is None:
        shift = 1

    if extmod is None:
        extmod = 'per'

    if type_ == 0 or type_ == 1:
        y = torch.from_numpy(resamp4c(x, type_, shift, extmod)).to(device)
    elif type_ == 2 or type_ == 3:
        y = torch.from_numpy(resamp4c(x.transpose(0, 1, 3, 2), type_ - 2, shift, extmod).transpose(0, 1, 3, 2)).to(device)
    else:
        print("The second input (type_) must be one of {0, 1, 2, 3}")

    return y

def ffilters(h0, h1, device=torch.device("cpu")):
    f0 = [[None]] * 4
    f1 = [[None]] * 4

    # For the first half channels
    f0[0] = modulate2(h0, 'r', None)
    f1[0] = modulate2(h1, 'r', None)

    f0[1] = modulate2(h0, 'c', None)
    f1[1] = modulate2(h1, 'c', None)

    # For the second half channels,
    # use the transposed filters of the first half channels
    f0[2] = f0[0].conj().T
    f1[2] = f1[0].conj().T

    f0[3] = f0[1].conj().T
    f1[3] = f1[1].conj().T

    return f0, f1

def backsamp(y, device=torch.device("cpu")):
    """ BACKSAMP
    Backsampling the subband images of the directional filter bank

       y = backsamp(y)

     Input and output are cell vector of dyadic length

     This function is called at the end of the DFBDEC to obtain subband images
     with overall sampling as diagonal matrices

     See also: DFBDEC"""

    # Number of decomposition tree levels
    n = int(log2(len(y)))

    if (n != round(n)) or (n < 1):
        print("Input must be a cell vector of dyadic length")
    if n == 1:
        # One level, the decomposition filterbank shoud be Q1r
        # Undo the last resampling (Q1r = R2 * D1 * R3)
        for k in range(0, 2):
            y[k] = resamp(y[k], 3, None, None, device=device)
            y[k][:, 0::2] = resamp(y[k][:, 0::2], 0, None, None, device=device)
            y[k][:, 1::2] = resamp(y[k][:, 1::2], 0, None, None, device=device)

    elif n > 2:
        N = 2**(n - 1)
        for k in range(0, 2**(n - 2)):
            shift = 2 * (k + 1) - (2**(n - 2) + 1)
            # The first half channels
            y[2 * k] = resamp(y[2 * k], 2, shift, None, device=device)
            y[2 * k + 1] = resamp(y[2 * k + 1], 2, shift, None, device=device)
            # The second half channels
            y[2 * k + N] = resamp(y[2 * k + N], 0, shift, None, device=device)
            y[2 * k + 1 + N] = resamp(y[2 * k + 1 + N], 0, shift, None, device=device)

    return y

def wfb2dec(x, h, g, device=torch.device("cpu")):
    """% WFB2DEC   2-D Wavelet Filter Bank Decomposition
    %
    %       y = wfb2dec(x, h, g)
    %
    % Input:
    %   x:      input image
    %   h, g:   lowpass analysis and synthesis wavelet filters
    %
    % Output:
    %   x_LL, x_LH, x_HL, x_HH:   Four 2-D wavelet subbands"""

    # Make sure filter in a row vector
    h = h[:, newaxis].reshape(len(h),)
    g = g[:, newaxis].reshape(len(g),)

    h0 = h
    len_h0 = len(h0)
    ext_h0 = floor(len_h0 / 2.0)
    # Highpass analysis filter: H1(z) = -z^(-1) G0(-z)
    len_h1 = len(g)
    c = floor((len_h1 + 1.0) / 2.0)
    # Shift the center of the filter by 1 if its length is even.
    if mod(len_h1, 2) == 0:
        c = c + 1
    # print(c)
    h1 = - g * (-1)**(arange(1, len_h1 + 1) - c)
    ext_h1 = len_h1 - c + 1

    # Row-wise filtering
    x_L = rowfiltering(x, h0, ext_h0)
    x_L = x_L[:, :, :, ::2]  # (:, 1:2:end)

    x_H = rowfiltering(x, h1, ext_h1)
    x_H = x_H[:, :, :, ::2]  # x_H(:, 1:2:end);

    # Column-wise filtering
    x_LL = rowfiltering(x_L.conj().permute(0, 1, 3, 2), h0, ext_h0)
    x_LL = x_LL.conj().permute(0, 1, 3, 2)
    x_LL = x_LL[:, :, ::2, :]

    x_LH = rowfiltering(x_L.conj().permute(0, 1, 3, 2), h1, ext_h1)
    x_LH = x_LH.conj().permute(0, 1, 3, 2)
    x_LH = x_LH[:, :, ::2, :]

    x_HL = rowfiltering(x_H.conj().permute(0, 1, 3, 2), h0, ext_h0)
    x_HL = x_HL.conj().permute(0, 1, 3, 2)
    x_HL = x_HL[:, :, ::2, :]

    x_HH = rowfiltering(x_H.conj().permute(0, 1, 3, 2), h1, ext_h1)
    x_HH = x_HH.conj().permute(0, 1, 3, 2)
    x_HH = x_HH[:, :, ::2, :]

    return x_LL, x_LH, x_HL, x_HH

def rowfiltering(x, f, ext1, device=torch.device("cpu")):
    """Internal function: Row-wise filtering with border handling"""
    
    ext1 = int(ext1)
    ext2 = int(len(f) - ext1 - 1)
    x = np.concatenate([x[:, :, :, -ext1::], x, x[:, :, :, 0:ext2]], axis=3)
    
    inputs = torch.from_numpy(x.astype(np.float32)).conj().permute(0, 1, 3, 2)
    filters = torch.from_numpy(f[:, np.newaxis].astype(np.float32))
    filters = filters[None, None, :, :]
    filters = torch.repeat_interleave(filters, x.shape[1], axis=0)
    y = F.conv2d(input=inputs, weight=filters, groups=inputs.size(1)).conj().permute(0, 1, 3, 2)

    return y