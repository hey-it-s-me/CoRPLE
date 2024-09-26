cimport cython
import numpy as np
cimport numpy as np
import numpy.ma as ma
import copy

def is_string_like(obj):
    """Return True if *obj* looks like a string
    
    Ported from: https://github.com/pieper/matplotlib/blob/master/lib/matplotlib/cbook.py
    with modification to adapt to Python3.
    """
    if isinstance(obj, (str)):
        return True
    # numpy strings are subclass of str, ma strings are not
    if ma.isMaskedArray(obj):
        if obj.ndim == 0 and obj.dtype.kind in 'SU':
            return True
        else:
            return False
    try:
        obj + ''
    except:
        return False
    return True

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.cdivision           # Deactivate tests for division by zero
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef resampc_periodic(DTYPE_t[:, :, :, :] x, int m, int n, DTYPE_t[:, :, :, :] y, int type_, int shift, int batch, int channel):
    # Py_ssize_t is the proper C type for Python array indices.
    cdef Py_ssize_t i, j, k, b, c
    for b in range(batch):
        for c in range(channel):
            for j in range(n):
                # Circular shift in each column
                if type_ == 0:
                    k = (shift * j) % m
                else:
                    k = (-shift * j) % m

                # Convert to non-negative mod if needed
                if k < 0:
                    k += m
                for i in range(m):
                    if k >= m:
                        k -= m

                    y[b, c, i, j] = x[b, c, k, j]

                    k += 1
    
    return y
    
def resamp4c(x, type_, shift, extmod):
    """RESAMPC. Resampling along the column

    y = resampc(x, type, shift, extmod)

    Input:
        x:      image that is extendable along the column direction
        type:   either 0 or 1 (0 for shuffering down and 1 for up)
        shift:  amount of shifts (typically 1)
        extmod: extension mode:
         - 'per': periodic
         - 'ref1': reflect about the edge pixels
         - 'ref2': reflect, doubling the edge pixels

    Output:
        y: resampled image with:
           R1 = [1, shift; 0, 1] or R2 = [1, -shift; 0, 1]
    """
    if type_ != 0 and type_ != 1:
        print("The second input (type_) must be either 1 or 2")
        return
    
    if shift == 0:
        print("The third input (shift) cannot be 0")
        return
    
    if is_string_like(extmod) != 1:
        print("EXTMOD arg must be a string")
        return
    
    y = np.zeros(x.shape, dtype=DTYPE)
    
    # The "cdef" keyword is also used within functions to type variables. It
    # can only be used at the top indentation level (there are non-trivial
    # problems with allowing them in other places, though we'd love to see
    # good and thought out proposals for it).
   
    cdef Py_ssize_t batch = x.shape[0]
    cdef Py_ssize_t channel = x.shape[1]
    cdef Py_ssize_t m = x.shape[2]
    cdef Py_ssize_t n = x.shape[3]
    
    assert x.shape == y.shape
    assert x.dtype == DTYPE
    assert y.dtype == DTYPE
    
    if extmod == 'per':
        resampc_periodic(x, m, n, y, type_, shift, batch, channel)
                
    return y