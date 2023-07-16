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

def distance_transform(src : np.ndarray):
    return cv2.distanceTransform(src, cv2.DIST_L2, 5)

def read_png(filepath):
    # return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    # return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(filepath, 0) # GRAYSCALE uint8

def compute_sdf(src : np.ndarray):
    return distance_transform(src) - distance_transform(255-src)

def load_data(datatype = 'test'):
    file = f'/mnt/data1/cjh/deepvecfont/data/vecfont_dataset/{datatype}/{datatype}_all.pkl'
    with open(file, 'rb') as f:
        data = pickle.load(f)
        print('finish load dataset')
        return data

# with open('/mnt/data1/cjh/deepvecfont/data/vecfont_dataset/train/train_all.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print('finish load dataset')

# print(len(data))
# exit()

def len_CubicBezier(cx0, cy0, cx1, cy1, cx2, cy2, cx3, cy3):
    # cb0 = (1-t)**3
    # cb1 = 3*t*(1-t)**2
    # cb2 = 3*(1-t)*t**2
    # cb3 = t**3
    # x = cx0*cb0 + cx1*cb1 + cx2*cb2 + cx3*cb3
    # y = cy0*cb0 + cy1*cb1 + cy2*cb2 + cy3*cb3
    f = lambda t:np.math.sqrt(
        ((-3*(1-t)**2)*cx0 + 3*(3*t**2-4*t+1)*cx1 + 3*(2*t-3*t**2)*cx2 + (3*t**2)*cx3)**2 + 
        ((-3*(1-t)**2)*cy0 + 3*(3*t**2-4*t+1)*cy1 + 3*(2*t-3*t**2)*cy2 + (3*t**2)*cy3)**2
    )
    return integrate.quad(f, 0, 1)

def tangent_CubicBezier(cx0, cy0, cx1, cy1, cx2, cy2, cx3, cy3, t):
    tangent_x = (-3*(1-t)**2)*cx0 + 3*(3*t**2-4*t+1)*cx1 + 3*(2*t-3*t**2)*cx2 + (3*t**2)*cx3
    tangent_y = (-3*(1-t)**2)*cy0 + 3*(3*t**2-4*t+1)*cy1 + 3*(2*t-3*t**2)*cy2 + (3*t**2)*cy3
    return unitize(tangent_x, tangent_y)

def unitize(x, y):
    modlen = np.sqrt(x*x+y*y)
    return x/modlen, y/modlen

def normal_CubicBezier(cx0, cy0, cx1, cy1, cx2, cy2, cx3, cy3, t):
    gradx = ((-3*(1-t)**2)*cx0 + 3*(3*t**2-4*t+1)*cx1 + 3*(2*t-3*t**2)*cx2 + (3*t**2)*cx3)
    grady = ((-3*(1-t)**2)*cy0 + 3*(3*t**2-4*t+1)*cy1 + 3*(2*t-3*t**2)*cy2 + (3*t**2)*cy3)
    return unitize(-grady, gradx)

def normal_Line(lx0, ly0, lx1, ly1):
    dx = lx1 - lx0
    dy = ly1 - ly0
    return unitize(-dy, dx)

def len_Line(lx0, ly0, lx1, ly1):
    # x = t * (lx1 - lx0) + lx0
    # y = t * (ly1 - ly0) + ly0
    return np.math.sqrt((lx0-lx1)**2+(ly0-ly1)**2)

def sample_svg(svgparam, svglen, nSample, render_offset=[0,-144/1024.], scaleX = 1., scaleY = 1.):
    svgparam = np.array(svgparam)
    svgparam = svgparam.reshape(-1, 10)
    svgparam = svgparam[0:svglen]
    render_offset[0] += 0.5*(1-scaleX)
    render_offset[1] += 0.5*(1-scaleY)
    cur = np.array(render_offset).astype(np.float32)
    svlenarr = np.zeros(svglen, np.float32)
    svlen = 0
    result = np.zeros((nSample, 2), dtype=np.float32)
    normal = np.zeros((nSample, 2), dtype=np.float32)
    alphai = 0
    for j, svgp in enumerate(svgparam):
        for i in range(4,10):
            svgp[i] /= 24. # normalization and acceleration
            svgp[i] *= scaleX if (i % 2 == 0) else scaleY
        if svgp[1] > 0:
            cur += np.array([svgp[8], svgp[9]])
        elif svgp[2] > 0:
            cur1 = cur + np.array([svgp[8], svgp[9]])
            l = len_Line(cur[0], cur[1], cur1[0], cur1[1])
            svlen += l
            svlenarr[j] = l
            cur = cur1
        elif svgp[3] > 0:
            cur1 = cur + np.array([svgp[4], svgp[5]])
            cur2 = cur + np.array([svgp[6], svgp[7]])
            cur3 = cur + np.array([svgp[8], svgp[9]])
            l, _ = len_CubicBezier(cur[0], cur[1], cur1[0], cur1[1], cur2[0], cur2[1], cur3[0], cur3[1])
            svlen += l
            svlenarr[j] = l
            cur = cur3
    svlenarr /= svlen
    cur = np.array(render_offset).astype(np.float32)
    samples_count = 0
    for i, svgp in enumerate(svgparam):
        if svgp[1] > 0:
            cur += np.array([svgp[8], svgp[9]])
        elif svgp[2] > 0:
            cur1 = cur + np.array([svgp[8], svgp[9]])
            alphaj = svlenarr[i] + alphai
            if (i == svglen - 1):
                alphaj = 1.
            samples = int(alphaj * nSample) - int(alphai * nSample)
            alphai = alphaj
            if(samples == 0):
                continue
            t = np.linspace(0, 1-1./samples, samples)
            x = t * (cur1[0] - cur[0]) + cur[0]
            y = t * (cur1[1] - cur[1]) + cur[1]
            result[samples_count:(samples_count+samples), 0] = y
            result[samples_count:(samples_count+samples), 1] = x
            normx, normy = normal_Line(cur[0], cur[1], cur1[0], cur1[1])
            normal[samples_count:(samples_count+samples), 0] = normy
            normal[samples_count:(samples_count+samples), 1] = normx
            samples_count += samples
            cur = cur1
        elif svgp[3] > 0:
            cur1 = cur + np.array([svgp[4], svgp[5]])
            cur2 = cur + np.array([svgp[6], svgp[7]])
            cur3 = cur + np.array([svgp[8], svgp[9]])
            alphaj = svlenarr[i] + alphai
            if (i == svglen - 1):
                alphaj = 1.
            samples = int(alphaj * nSample) - int(alphai * nSample)
            alphai = alphaj
            if(samples == 0):
                continue
            t = np.linspace(0.01, 1-1./samples, samples)
            cb0 = (1-t)**3
            cb1 = 3*t*(1-t)**2
            cb2 = 3*(1-t)*t**2
            cb3 = t**3
            x = cur[0]*cb0 + cur1[0]*cb1 + cur2[0]*cb2 + cur3[0]*cb3
            y = cur[1]*cb0 + cur1[1]*cb1 + cur2[1]*cb2 + cur3[1]*cb3
            result[samples_count:(samples_count+samples), 0] = y
            result[samples_count:(samples_count+samples), 1] = x
            normx, normy = normal_CubicBezier(cur[0], cur[1], cur1[0], cur1[1], cur2[0], cur2[1], cur3[0], cur3[1], t)
            normal[samples_count:(samples_count+samples), 0] = normy
            normal[samples_count:(samples_count+samples), 1] = normx
            samples_count += samples
            cur = cur3
    return result, normal



def mat2path(svgparam, svglen, resolution = 1024, render_offset = [0,-144], scaleX = 1., scaleY = 1.):
    svgparam = np.array(svgparam)
    svgparam = svgparam.reshape(-1, 10)
    svgparam = svgparam[0:svglen]
    render_offset[0] += 512.*(1-scaleX)
    render_offset[1] += 512.*(1-scaleY)
    cur = np.array(render_offset).astype(np.float32)
    svgseq = ''
    # print(svgparam)
    for svgp in svgparam:
        for i in range(4,10):
            svgp[i] *= (resolution / 24.)
            svgp[i] *= scaleX if (i % 2 == 0) else scaleY
            # import pdb; pdb.set_trace()
        if svgp[0] > 0:
            pass
        elif svgp[1] > 0:
            svgseq += 'M '
            cur += np.array([svgp[8], svgp[9]])
            svgseq += f'{cur[0]} {cur[1]} '
        elif svgp[2] > 0:
            # cur1 = cur + np.float32(0.33) * np.array([svgp[8], svgp[9]]) - np.array([0, 100])
            # cur2 = cur + np.float32(0.66) * np.array([svgp[8], svgp[9]]) - np.array([0, 100])
            # curmid = cur + np.float32(0.5) * np.array([svgp[8], svgp[9]]) - np.array([0, 100])
            cur = cur + np.array([svgp[8], svgp[9]])
            # svgseq += f'C {cur1[0]} {cur1[1]} {cur2[0]} {cur2[1]} {cur[0]} {cur[1]} '
            # svgseq += f'Q {curmid[0]} {curmid[1]} {cur[0]} {cur[1]} '
            svgseq += f'L {cur[0]} {cur[1]} '
        elif svgp[3] > 0:
            svgseq += 'C '
            cur1 = cur + np.array([svgp[4], svgp[5]])
            cur2 = cur + np.array([svgp[6], svgp[7]])
            cur = cur + np.array([svgp[8], svgp[9]])
            svgseq += f'{cur1[0]} {cur1[1]} {cur2[0]} {cur2[1]} {cur[0]} {cur[1]} '
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{resolution}px" height="{resolution}px" style="-ms-transform: rotate(360deg); -webkit-transform: rotate(360deg); transform: rotate(360deg);" preserveAspectRatio="xMidYMid meet" viewBox="0 0 {resolution} {resolution}">'
    f'<path d="{svgseq}" stroke-width="1.0" fill="rgb(0, 0, 0)" opacity="1.0"></path></svg>')


def svg2png(data, folder, resolution):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in tqdm(range(len(data))):
        for j in range(52):
            fname = f'{i}_{j}.png'
            wrpath = os.path.join(folder, fname)
            y=mat2path(data[i]['sequence'][j], data[i]['seq_len'][j][0], resolution)
            cairosvg.svg2png(bytestring=y, write_to=wrpath
                ,background_color='#FFFFFFFF')
            # sdf = compute_sdf(read_png(wrpath))
            # cv2.imwrite(os.path.join(folder, f'{i}_{j}_sdf.png'), sdf)

def path2png(patharr, pathlen, f_output):
    y = mat2path(patharr, pathlen)
    print(y)
    cairosvg.svg2png(bytestring=y, write_to=f_output
                ,background_color='#FFFFFFFF')

if __name__ == '__main__':
    pass




