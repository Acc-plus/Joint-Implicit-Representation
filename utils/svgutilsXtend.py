import cairosvg
import cv2
import numpy as np
import svgwrite
import svgpathtools
import pickle
import time
import os
from tqdm import tqdm
import torch
import scipy.integrate as integrate
from utils.svgutils import *

def gen_corners(svgparam, svglen, render_offset=[0,-144/1024.], scaleX = 1., scaleY = 1.):
    svgparam = np.array(svgparam)
    svgparam = svgparam.reshape(-1, 10)
    svgparam = svgparam[0:svglen]
    render_offset[0] += 0.5*(1-scaleX)
    render_offset[1] += 0.5*(1-scaleY)
    cur = np.array(render_offset).astype(np.float32)

    record_hdr = np.zeros((2, ), dtype=np.float32)
    record_begin = np.zeros((2, ), dtype=np.float32)
    record_end = np.zeros((2, ), dtype=np.float32)
    corners = []
    record_sign = False
    # import pdb; pdb.set_trace()
    for j, svgp in enumerate(svgparam):
        # import pdb; pdb.set_trace()
        for i in range(4,10):
            svgp[i] /= 24. # normalization and acceleration
            svgp[i] *= scaleX if (i % 2 == 0) else scaleY
        if svgp[1] > 0:
            costheta = record_hdr[0] * record_end[0] + record_hdr[1] * record_end[1]
            if (np.abs(record_hdr).sum() > 1e-6 and 1. - costheta > 0.01):
                corners.append(cur[:]) # a new corner
            record_sign = True
            record_end[:] = 0.
            cur += np.array([svgp[8], svgp[9]])
        elif svgp[2] > 0:
            cur1 = cur + np.array([svgp[8], svgp[9]])
            dx, dy = unitize(svgp[8], svgp[9])
            record_begin[0] = dx
            record_begin[1] = dy
            costheta = record_begin[0] * record_end[0] + record_begin[1] * record_end[1]
            if (np.abs(record_end).sum() > 1e-6 and 1. - costheta > 0.01):
                corners.append(cur[:]) # a new corner
            record_end[0] = dx
            record_end[1] = dy
            if (record_sign):
                record_hdr[:] = record_begin[:]
                record_sign = False
            cur = cur1
        elif svgp[3] > 0:
            cur1 = cur + np.array([svgp[4], svgp[5]])
            cur2 = cur + np.array([svgp[6], svgp[7]])
            cur3 = cur + np.array([svgp[8], svgp[9]])
            tx, ty = tangent_CubicBezier(cur[0], cur[1], cur1[0], cur1[1], cur2[0], cur2[1], cur3[0], cur3[1], 0.01)
            record_begin[0] = tx
            record_begin[1] = ty
            costheta = record_begin[0] * record_end[0] + record_begin[1] * record_end[1]
            if (np.abs(record_end).sum() > 1e-6 and 1. - costheta > 0.01):
                corners.append(cur[:]) # a new corner
            tx, ty = tangent_CubicBezier(cur[0], cur[1], cur1[0], cur1[1], cur2[0], cur2[1], cur3[0], cur3[1], 0.99)
            record_end[0] = tx
            record_end[1] = ty
            if (record_sign):
                record_hdr[:] = record_begin[:]
                record_sign = False
            cur = cur3

    costheta = record_hdr[0] * record_end[0] + record_hdr[1] * record_end[1]
    if (np.abs(record_hdr).sum() > 1e-6 and 1. - costheta > 0.01):
        corners.append(cur[:]) # a new corner
    return np.array(corners).astype(np.float32)