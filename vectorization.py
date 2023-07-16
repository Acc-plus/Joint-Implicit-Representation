from models.DoubleIFGAN import DoubleImplicitField
from models.sdf import ImplicitField
from utils.dataloader import *
from models.icf import *
import torch
import torch.nn as nn
import random
import numpy as np
import yaml
from globalconfig import *
from datetime import *
import cv2
import imageio
# from utils.combrender import Vectorize_Image
from utils.svgvectorizer import Vectorize_Image
import cairosvg
from img_collect import ImgCollection
model_toload = configs['log_info']['pretrained']
num_instance = configs['data']['num_instance']
data_path = configs['data']['path']
corner_path = os.path.join(data_path, 'corners')
gif_save = configs['log_info']['gif_save']
png_save = configs['log_info']['png_save']
epoch = configs['log_info']['epoch']

def vectorize(x):
    print(f'vectorization {x}')
    f_img = f'results/{model_toload}/{x}.png'
    f_corner = f'results/{model_toload}/{x}c.npy'
    Vectorize_Image(f_img, f_corner, f'results/{model_toload}/{x}.svg')

glyph_map = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15,
    'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25, 
    'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33,
    'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41,
    'q': 42, 'r': 43, 's': 44, 't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49,
    'y': 50, 'z': 51, 
}

rev_gm = dict()
for gl in glyph_map.keys():
    rev_gm[glyph_map[gl]] = gl

def render_group(x):
    # x = 118
    for alpha in glyph_map.keys():
        f_img = f'renders/{alpha}/{x}.png'
        f_corner = f'renders/{alpha}c/{x}.npy'
        # Vectorize_Image(f_img, f_corner, f'{x}.svg', 512)
        if not os.path.exists(f'./svg/{x:04d}'):
            os.makedirs(f'./svg/{x:04d}')
        strsvg = Vectorize_Image(f_img, f_corner, f'./svg/{x:04d}/{glyph_map[alpha]}.svg', 512)
        cairosvg.svg2png(bytestring=strsvg, write_to=f'./svg/{x:04d}/{glyph_map[alpha]}.png', background_color='#FFFFFF')

    collector = ImgCollection()
    # x = 32
    for alpha in glyph_map.keys():
        collector.add(f'./svg/{x:04d}/{glyph_map[alpha]}.png')
        
    collector.render_img(f'./svg/{x:04d}/all.png')

glyph_map = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15,
    'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25, 
    'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33,
    'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41,
    'q': 42, 'r': 43, 's': 44, 't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49,
    'y': 50, 'z': 51, 
}

mapping_dict = dict()
for gl in glyph_map.keys():
    mapping_dict[glyph_map[gl]] = gl

def render_whole():
    for idx in range(52):
        if not os.path.exists(f'svg/{rev_gm[idx]}'):
            os.makedirs(f'svg/{rev_gm[idx]}')
        print(f'svg/{rev_gm[idx]}')
        for x in tqdm(range(1425)):
            # if os.path.exists(f'svg/{rev_gm[idx]}/{x}.svg'):
            #     continue
            try:
                f_img = f'renders/{rev_gm[idx]}/{x}.png'
                f_corner = f'renders/{rev_gm[idx]}c/{x}.npy'
                # import pdb;pdb.set_trace()
                svgp = Vectorize_Image(f_img, f_corner, f'svg/{rev_gm[idx]}/{x}.svg', 512)
                cairosvg.svg2png(bytestring=svgp, write_to=f'svg/{rev_gm[idx]}/{x}.png' ,background_color='#FFFFFFFF')
            except:
                print(f'svg/{rev_gm[idx]}/{x}.svg')
                pass


def interpolate_whole():
    for idx in range(52):
        if not os.path.exists(f'isvg/{rev_gm[idx]}'):
            os.makedirs(f'isvg/{rev_gm[idx]}')
        print(f'svg/{rev_gm[idx]}')
        for x in tqdm(range(500)):
            f_img = f'interpolation/{rev_gm[idx]}/{x}.png'
            f_corner = f'interpolation/{rev_gm[idx]}c/{x}.npy'
            # import pdb;pdb.set_trace()
            svgp = Vectorize_Image(f_img, f_corner, f'isvg/{rev_gm[idx]}/{x}.svg', 512)
            cairosvg.svg2png(bytestring=svgp, write_to=f'isvg/{rev_gm[idx]}/{x}.png' ,background_color='#FFFFFFFF')

def vectoriza_demo(path):
    for i in range(5):
        f_img = os.path.join(path, f'{i}.png')
        f_corner = os.path.join(path, f'{i}.npy')
        strsvg = Vectorize_Image(f_img, f_corner, os.path.join(path, f'{i}.svg'), 512)
        cairosvg.svg2png(bytestring=strsvg, write_to=os.path.join(path, f'{i}v.png'), background_color='#FFFFFF')

    collector = ImgCollection()
    # x = 32
    for i in range(5):
        collector.add(os.path.join(path, f'{i}v.png'))
        
    collector.render_img(os.path.join(path, f'all.png'))
def vectoriza_demo_all(font1, font2):
    num_render = 52
    path = f'demo/{font1}_{font2}'
    for alpha in tqdm(range(num_render)):
        for i in range(5):
            f_img = os.path.join(path, f'{alpha}_{i}.png')
            f_corner = os.path.join(path, f'{alpha}_{i}.npy')
            strsvg = Vectorize_Image(f_img, f_corner, os.path.join(path, f'{alpha}_{i}.svg'), 512)
            # cairosvg.svg2png(bytestring=strsvg, write_to=os.path.join(path, f'{alpha}_{i}v.png'), background_color='#FFFFFF')

    # collector = ImgCollection()
    # # x = 32
    # collector.resolution = 32
    # for alpha in range(num_render):
    #     for i in range(5):
    #         collector.add(os.path.join(path, f'{alpha}_{i}v.png'))
        
    # collector.render_img(os.path.join(path, f'all.png'))

if __name__ == '__main__':
    # Vectorize_Image('/mnt/data1/cjh/ncf/results/SDF_52_VL/89_964_0.3.png', '/mnt/data1/cjh/ncf/89_964_03.npy', f'reb.svg', 512)
    for i in range(26):
        Vectorize_Image(f'rebuttal/{i}.png', f'rebuttal/x{i}.npy', f'{i}.svg', 512)
    # interpolate_whole()
    # render_whole()
    # f_img = f'results/sdf_A/1436_1321_0.0.png'
    # f_corner = [f'results/corner_sdflowLarge/1436_1321_0.0c.npy']
    # f_img = f'results/sdf_multi_26_test_lrd995/12.png'
    # f_corner = [f'results/cf_multi_26_test_lrd995_dv/12c.npy']
    # for x in range(1425):
    #     try:
    #         f_img = f'evaluation/A/{x}.png'
    #         f_corner = f'evaluation/Ac/{x}.npy'
    #         Vectorize_Image(f_img, f_corner, f'svg/A/{x}.svg', 512)
    #     except:
    #         pass
    # f_img = f'evaluation/A/{x}.png'
    # f_corner = f'evaluation/Ac/{x}.npy'
    
    # render_group(32)
    # render_group(0)
    # render_group(14)
    # render_group(999)
    # render_group(95)
    # vectoriza_demo_all(0, 1076)
    # render_whole()
    # x = 15
    # alpha = 'C'
    # f_img = f'renders/{alpha}/{x}.png'
    # f_corner = f'renders/{alpha}c/{x}.npy'
    # Vectorize_Image(f_img, f_corner, f'{x}.svg', 512)
    # render_whole()
    # f_img = f'mid.png'
    # f_corner = f'mid.npy'
    # # f_corner = f'results/CF_52_lvlv/999c.npy'
    # Vectorize_Image(f_img, f_corner, f'mid.svg', 512)
    # vectoriza_demo('demo/g8')
    # vectoriza_demo_all('demo/0_201')
    # vectoriza_demo_all('demo/0_26')
    # vectoriza_demo_all(0, 341)
    # vectoriza_demo_all(1076, 999)
    # vectoriza_demo_all(999, 1117)
    # vectoriza_demo_all(1117, 1103)

    # f_img = f'demo/0_341/17_3.png'
    # f_corner = f'results/CF_52_lvlv/0c.npy'
    # Vectorize_Image(f_img, f_corner, f'output.svg', 512)




