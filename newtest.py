import numpy as np
import interval
from interval import get_lu, get_iarray, from_cent_pert
from ReachMM.utils import d_metzler, d_positive

i = np.interval(1,0.1)
I = np.array([[i, i*2],[i**2, 4*i]])
Il, Iu = get_lu(I)
print(I)
print(Il, Iu)

Ilp, Iln = d_positive(Il)
Iup, Iun = d_positive(Iu)

x = from_cent_pert([1,1],[2,2])
xl, xu = get_lu(x)

rl = Ilp @ xl + Iln @ xu
ru = Iup @ xu + Iun @ xl
r = get_iarray(rl, ru)
print(r)
print(I @ x)
