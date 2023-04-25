# from numpy import _sin, _cos, abs, pi, _sign, inf, diag_indices_from, clip, empty
from numpy import sin as _sin
from numpy import cos as _cos
from numpy import sign as _sign
from numpy import pi

class interval :
    pass

class iarray :
    pass

def sin(_xx_) :
    n = len(_xx_) // 2
    _x, x_ = _xx_[:n], _xx_[n:]
    cx = _cos(_x)
    cxhat = _cos(x_)
    absdiff = abs(_x - x_)
    # if absdiff >= 2*pi :
    #     return _sign(_x - x_)
    # if cx <= 0 and 0 <= cxhat:
    #     return _sign(_x - x_)
    # if absdiff <= pi :
    #     if cx >= 0 and cxhat >= 0 :
    #         return _sin(_x)
    #     if cx <= 0 and cxhat <= 0 :
    #         return _sin(x_)
    # if _x <= x_ and cx >= 0 and 0 >= cxhat :
    #     return min(_sin(_x), _sin(x_))
    # if _x >= x_ and cx >= 0 and 0 >= cxhat :
    #     return max(_sin(_x), _sin(x_))
    # return _sign(_x - x_)

    if cx >= 0 and cxhat >= 0 and absdiff <= pi :
        return _sin(_x)
    if cx <= 0 and cxhat <= 0 and absdiff <= pi :
        return _sin(x_)
    if absdiff >= 2*pi :
        return _sign(_x - x_)
    if cx <= 0 and 0 <= cxhat and absdiff <= 2*pi :
        return _sign(_x - x_)
    if cx*cxhat >= 0 and pi <= absdiff and absdiff <= 2*pi :
        return _sign(_x - x_)
    if _x <= x_ and cx >= 0 and 0 >= cxhat and absdiff <= 2*pi :
        return min(_sin(_x), _sin(x_))
    if _x >= x_ and cx >= 0 and 0 >= cxhat and absdiff <= 2*pi :
        return max(_sin(_x), _sin(x_))

def cos (_xx_) :
    # n = len(_xx_) // 2
    # _x, x_ = _xx_[:n], _xx_[n:]
    # return d_sin(_x + pi/2, x_ + pi/2)
    return sin(_xx_ + pi/2)

def d_b1b2(b, bhat) :
    values = (b[0]*b[1], bhat[0]*b[1], b[0]*bhat[1], bhat[0]*bhat[1])
    return min(values) if b[0] < bhat[0] else max(values)
    # if b[0] < bhat[0] :
    #     return min(values)
    # else :
    #     return max(values)

def d_x2(_x, x_) :
    if _x >= 0 and _x >= -x_ :
        return _x**2
    elif x_ <= 0 and _x <= -x_ :
        return x_**2
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
