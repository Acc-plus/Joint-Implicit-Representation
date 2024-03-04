import numpy as np
from tqdm import tqdm
import scipy.integrate as integrate

def reshape_params(lmost, rmost, umost, dmost, coe = 1.2):
    # if (rmost - lmost < 0.9) and (umost - dmost < 0.9):
    #     trans_lr = 0.5 - (rmost + lmost)/2
    #     trans_ud = 0.5 - (umost + dmost)/2
    #     scale = 1.
    # else:
    scale = max((rmost - lmost), (dmost - umost)) * coe
    if (np.abs(scale) <= 0.001):
        return 0., 0., 1.
    rmost /= scale
    lmost /= scale
    umost /= scale
    dmost /= scale
    trans_lr = 0.5 - (rmost + lmost)/2
    trans_ud = 0.5 - (umost + dmost)/2
    # print(lmost, rmost, umost, dmost)
    return trans_lr, trans_ud, scale

def reshape_svg(svgparam, svglen, coe = 1.2):
    svgparam = np.array(svgparam)
    result = np.array(svgparam)
    svgparam = svgparam.reshape(-1, 10)
    svgparam = svgparam[0:svglen]
    cur = np.array([0, 0]).astype(np.float32)
    
    lmost = 1.
    rmost = 0.
    umost = 1.
    dmost = 0.
    for j, svgp in enumerate(svgparam):
        for i in range(4,10):
            svgp[i] /= 24. # normalization and acceleration
        if svgp[1] > 0:
            cur += np.array([svgp[8], svgp[9]])
        elif svgp[2] > 0:
            cur1 = cur + np.array([svgp[8], svgp[9]])
            cur = cur1
        elif svgp[3] > 0:
            cur1 = cur + np.array([svgp[4], svgp[5]])
            cur2 = cur + np.array([svgp[6], svgp[7]])
            cur3 = cur + np.array([svgp[8], svgp[9]])
            cur = cur3
        lmost = min(lmost, cur[0])
        rmost = max(rmost, cur[0])
        umost = min(umost, cur[1])
        dmost = max(dmost, cur[1])
    trans_lr, trans_ud, scale = reshape_params(lmost, rmost, umost, dmost, coe = coe)
    result /= scale
    # for i in range(4,10):
    #     result[i] += trans_lr if i % 2 == 0 else trans_ud
    # result *= 24.
    return result, trans_lr, trans_ud