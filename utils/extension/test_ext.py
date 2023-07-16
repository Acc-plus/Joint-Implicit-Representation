import enum
from os import dup
import torch
import numpy as np
import math
from scipy.special import comb as nOk
from numpy import linalg, matrix
import cv2

def Bernstein(n, t, k):
    return t**(k)*(1-t)**(n-k)*nOk(n,k)
def Qbezier(ts):
    return matrix([[Bernstein(2,t,k) for k in range(3)] for t in ts])

vqueue = []
vqueue.append(np.random.rand(2))
vqueue.append(np.random.rand(2))
vqueue.append(np.random.rand(2))

sample_t = np.linspace(0, 1, vqueue.__len__() * 2)
bezierM = Qbezier(sample_t)

import vectorization as vct

vct.chamfer_bezier(vqueue, bezierM)
print(vqueue)
print(bezierM)