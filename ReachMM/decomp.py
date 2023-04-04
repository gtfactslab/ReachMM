from numpy import sin, cos, abs, pi, sign, inf, diag_indices_from, clip, empty

def d_sin(x, xhat) :
    cx = cos(x)
    cxhat = cos(xhat)
    absdiff = abs(x - xhat)
    # if absdiff >= 2*pi :
    #     return sign(x - xhat)
    # if cx <= 0 and 0 <= cxhat:
    #     return sign(x - xhat)
    # if absdiff <= pi :
    #     if cx >= 0 and cxhat >= 0 :
    #         return sin(x)
    #     if cx <= 0 and cxhat <= 0 :
    #         return sin(xhat)
    # if x <= xhat and cx >= 0 and 0 >= cxhat :
    #     return min(sin(x), sin(xhat))
    # if x >= xhat and cx >= 0 and 0 >= cxhat :
    #     return max(sin(x), sin(xhat))
    # return sign(x - xhat)

    if cx >= 0 and cxhat >= 0 and absdiff <= pi :
        return sin(x)
    if cx <= 0 and cxhat <= 0 and absdiff <= pi :
        return sin(xhat)
    if absdiff >= 2*pi :
        return sign(x - xhat)
    if cx <= 0 and 0 <= cxhat and absdiff <= 2*pi :
        return sign(x - xhat)
    if cx*cxhat >= 0 and pi <= absdiff and absdiff <= 2*pi :
        return sign(x - xhat)
    if x <= xhat and cx >= 0 and 0 >= cxhat and absdiff <= 2*pi :
        return min(sin(x), sin(xhat))
    if x >= xhat and cx >= 0 and 0 >= cxhat and absdiff <= 2*pi :
        return max(sin(x), sin(xhat))

def d_cos (x, xhat) :
    return d_sin(x + pi/2, xhat + pi/2)

def d_b1b2(b, bhat) :
    values = (b[0]*b[1], bhat[0]*b[1], b[0]*bhat[1], bhat[0]*bhat[1])
    return min(values) if b[0] < bhat[0] else max(values)
    # if b[0] < bhat[0] :
    #     return min(values)
    # else :
    #     return max(values)

def d_x2(x, xhat) :
    if x >= 0 and x >= -xhat :
        return x**2
    elif xhat <= 0 and x <= -xhat :
        return xhat**2
    return 0

def d_metzler (A, separate=False)  :
    diag = diag_indices_from(A)
    Am = clip(A, 0, inf); Am[diag] = A[diag]
    An = A - Am
    if separate :
        return Am, An
    else :
        n = A.shape[0]
        ret = empty((2*n,2*n))
        ret[:n,:n] = Am; ret[n:,n:] = Am
        ret[:n,n:] = An; ret[n:,:n] = An
        return ret

def d_positive (B, separate=False) :
    Bp = clip(B, 0, inf); Bn = clip(B, -inf, 0)
    if separate :
        return Bp, Bn
    else :
        n,m = B.shape
        ret = empty((2*n,2*m))
        ret[:n,:m] = Bp; ret[n:,m:] = Bp
        ret[:n,m:] = Bn; ret[n:,:m] = Bn
        return ret
