from models.MCF import MultiCF
from models.MCF_EX import MultiCFEX
from utils.dataloader import *
import torch
import torch.nn as nn
import random
import numpy as np
import yaml
from globalconfig import *
from datetime import *
import cv2
import imageio

model_toload = configs['log_info']['pretrained']
num_instance = configs['data']['num_instance']
data_paths = configs['data']['paths']
gif_save = configs['log_info']['gif_save']
png_save = configs['log_info']['png_save']
epoch = configs['log_info']['epoch']
model_type = eval(configs['model']['type'])
ep = ''
if (epoch is not None):
    ep = f' ep{epoch}'

model = model_type(num_instance, len(data_paths), **configs['model']['params'])
assert model_toload is not None
model.load_state_dict(torch.load(f'results/{model_toload}/model{ep}.pth', map_location=device))

model = model.to(device)

output_dir = os.path.join('results', model_toload)

def inference_between(c, x, y, num_frames, suf = None):
    return model.inference_multi(c, x, y, num_frames, suf)

def inference_1(c, x, suf = None):
    return model.inference(c, x, suf)

def inference_mid(c, x, y, coe, mode = None):
    return model.inference_mid(c, x, y, coe)

resol = 512
resolution_mat = torch.FloatTensor([[[i, j] for j in range(resol)] for i in range(resol)]).to(device) / resol

def inf_cf(x, resolution=1024, output_file=None, level = 0.5):
    output_hm = f'results/{model_toload}/{png_save}{x}hm.png'
    if output_file is None:
        output_file = f'results/{model_toload}/{png_save}{x}c.npy'
        print(f'generate the font id {x}')
    im = np.zeros((resolution, resolution), dtype=np.uint8)
    imtorch = torch.zeros(resolution, resolution, device=device)
    coords = resolution_mat.reshape(-1, 2)
    sdf_output = inference_1(coords, x, 'cf')
    imtorch[:] = sdf_output.view(resolution, resolution).transpose(0, 1)
    # sdf_output = (sdf_output).view(num_frames, -1).cpu().numpy()
    sdf_output *= 255.
    sdf_output[sdf_output > 255] = 255
    im = (sdf_output).view(resolution, resolution).cpu().numpy().astype(np.uint8)
    
    # imheat = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    

    # imtorch = torch.zeros((resolution, resolution), device=device)
    imdif = torch.zeros(8, resolution, resolution, device=device)
    imdif[:] = imtorch
    imdif[0, 1:(resolution-1), 1:(resolution-1)] -= imtorch[:(resolution-2), :(resolution-2)]
    imdif[1, 1:(resolution-1), 1:(resolution-1)] -= imtorch[1:(resolution-1), :(resolution-2)]
    imdif[2, 1:(resolution-1), 1:(resolution-1)] -= imtorch[2:(resolution), :(resolution-2)]
    imdif[3, 1:(resolution-1), 1:(resolution-1)] -= imtorch[2:(resolution), 1:(resolution-1)]
    imdif[4, 1:(resolution-1), 1:(resolution-1)] -= imtorch[2:(resolution), 2:(resolution)]
    imdif[5, 1:(resolution-1), 1:(resolution-1)] -= imtorch[1:(resolution-1), 2:(resolution)]
    imdif[6, 1:(resolution-1), 1:(resolution-1)] -= imtorch[:(resolution-2), 2:(resolution)]
    imdif[7, 1:(resolution-1), 1:(resolution-1)] -= imtorch[:(resolution-2), 1:(resolution-1)]
    imb = imdif >= 0.
    maxima = imb[0] & imb[1] & imb[2] & imb[3] & imb[4] & imb[5] & imb[6] & imb[7] & (imtorch > level)

    
    # import pdb; pdb.set_trace()

    
    # for i in range(resolution):
    #     for j in range(resolution):
    #         if (maxima[i, j]):
    #             print(imtorch[i, j])
    #             cv2.circle(imheat, [i, j], 3, [0, 0, 0], 3)

    # import pdb; pdb.set_trace()
    maxima = resolution_mat[maxima.view(resolution, resolution)].cpu().numpy()
    # print(maxima.shape)
    # cv2.imwrite(f'results/{model_toload}/{png_save}{x}H.png', imheat)
    np.save(output_file, np.array(maxima))
    # imheat = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    # cv2.imwrite(output_hm, imheat)
    return im

def inf_cfs(x, y, num_frames=20):
    print(f'generate the font id {x}-{y}')
    ims = [np.zeros((1024, 1024), dtype=np.uint8) for _ in range(num_frames)]
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_between(coords, x, y, num_frames, 'cf')
        # sdf_output = (sdf_output).view(num_frames, -1).cpu().numpy()
        sdf_output *= 255.
        sdf_output[sdf_output > 255] = 255
        sdf_output = (sdf_output).view(num_frames, -1).cpu().numpy().astype(np.uint8)
        for j in range(num_frames):
            ims[j][i, :] = sdf_output[j]
    
    imheats = [cv2.applyColorMap(ims[i], cv2.COLORMAP_JET) for i in range(num_frames)]
    imageio.mimsave(f'results/{model_toload}/{gif_save}{x}_{y}.gif', imheats, 'GIF', duration=0.2)
    return ims


def inf_cf_mid(x, y, resolution, coe = 0.5, output_file = None):
    output_hm = f'results/{model_toload}/{png_save}{x}{y}{coe}hm.png'
    if output_file is None:
        output_file = f'results/{model_toload}/{png_save}{x}c.npy'
        print(f'generate the font id {x}')
    im = np.zeros((resolution, resolution), dtype=np.uint8)
    imtorch = torch.zeros(resolution, resolution, device=device)
    coords = resolution_mat.reshape(-1, 2)
    cf_output = inference_mid(coords, x, y, coe, 'cf')
    imtorch[:] = cf_output.view(resolution, resolution).transpose(0, 1)
    # sdf_output = (sdf_output).view(num_frames, -1).cpu().numpy()
    cf_output *= 255.
    cf_output[cf_output > 255] = 255
    im = (cf_output).view(resolution, resolution).cpu().numpy().astype(np.uint8)
    
    imheat = cv2.applyColorMap(im, cv2.COLORMAP_JET)

    # imtorch = torch.zeros((resolution, resolution), device=device)
    imdif = torch.zeros(8, resolution, resolution, device=device)
    imdif[:] = imtorch
    imdif[0, 1:(resolution-1), 1:(resolution-1)] -= imtorch[:(resolution-2), :(resolution-2)]
    imdif[1, 1:(resolution-1), 1:(resolution-1)] -= imtorch[1:(resolution-1), :(resolution-2)]
    imdif[2, 1:(resolution-1), 1:(resolution-1)] -= imtorch[2:(resolution), :(resolution-2)]
    imdif[3, 1:(resolution-1), 1:(resolution-1)] -= imtorch[2:(resolution), 1:(resolution-1)]
    imdif[4, 1:(resolution-1), 1:(resolution-1)] -= imtorch[2:(resolution), 2:(resolution)]
    imdif[5, 1:(resolution-1), 1:(resolution-1)] -= imtorch[1:(resolution-1), 2:(resolution)]
    imdif[6, 1:(resolution-1), 1:(resolution-1)] -= imtorch[:(resolution-2), 2:(resolution)]
    imdif[7, 1:(resolution-1), 1:(resolution-1)] -= imtorch[:(resolution-2), 1:(resolution-1)]
    imb = imdif >= 0.
    maxima = imb[0] & imb[1] & imb[2] & imb[3] & imb[4] & imb[5] & imb[6] & imb[7] & (imtorch > 0.5)

    
    # import pdb; pdb.set_trace()

    # imheat = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    # for i in range(resolution):
    #     for j in range(resolution):
    #         if (maxima[i, j]):
    #             print(imtorch[i, j])
    #             cv2.circle(imheat, [i, j], 3, [0, 0, 0], 3)

    # import pdb; pdb.set_trace()
    maxima = resolution_mat[maxima.view(resolution, resolution)].cpu().numpy()
    # print(maxima.shape)
    # cv2.imwrite(f'results/{model_toload}/{png_save}{x}H.png', imheat)
    np.save(output_file, np.array(maxima))
    # cv2.imwrite(output_hm, imheat)
    return im



print('--rendering start--')
model.eval()

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
    for idx in range(0, 52):
        model.set_glyph_idx(idx)
        if not os.path.exists(f'renders/{mapping_dict[idx]}c'):
            os.makedirs(f'renders/{mapping_dict[idx]}c')
        print(f'{mapping_dict[idx]}c')
        for i in tqdm(range(1425)):
            inf_cf(i, 512, f'renders/{mapping_dict[idx]}c/{i}.npy')

def interpo_whole():
    import numpy as np
    sel_id = np.array(list(range(1425)), dtype=np.int64)
    np.random.seed(14)
    np.random.shuffle(sel_id)
    for idx in range(52):
        if not os.path.exists(f'interpolation/{mapping_dict[idx]}c'):
            os.makedirs(f'interpolation/{mapping_dict[idx]}c')
        model.set_glyph_idx(idx)
        print(f'render {mapping_dict[idx]}c')
        for i in tqdm(range(500)):
            idx1 = sel_id[i*2]
            idx2 = sel_id[i*2+1]
            inf_cf_mid(idx1, idx2, 512, 0.5, f'interpolation/{mapping_dict[idx]}c/{i}.npy')

def interpo_sdf():
    import numpy as np
    sel_id = np.array(list(range(1425)), dtype=np.int64)
    np.random.seed(14)
    np.random.shuffle(sel_id)
    for idx in range(52):
        if not os.path.exists(f'abls/{mapping_dict[idx]}'):
            os.makedirs(f'abls/{mapping_dict[idx]}')
        model.set_glyph_idx(idx)
        print(f'render {mapping_dict[idx]}')
        for i in tqdm(range(500)):
            idx1 = sel_id[i*2]
            idx2 = sel_id[i*2+1]
            inf_cf_mid(idx1, idx2, 512, 0.5, f'abls/{mapping_dict[idx]}/{i}_sdf.npy')

def interpo_peak():
    for idx in range(0, 52):
        # print(idx)
        model.set_glyph_idx(idx)
        if not os.path.exists(f'ablp/{mapping_dict[idx]}c3p'):
            os.makedirs(f'ablp/{mapping_dict[idx]}c3p')
        if not os.path.exists(f'ablp/{mapping_dict[idx]}c7p'):
            os.makedirs(f'ablp/{mapping_dict[idx]}c7p')
        if not os.path.exists(f'ablp/{mapping_dict[idx]}c9p'):
            os.makedirs(f'ablp/{mapping_dict[idx]}c9p')
        print(f'{mapping_dict[idx]}c')
        for i in tqdm(range(1425)):
            inf_cf(i, 512, f'ablp/{mapping_dict[idx]}c3p/{i}.npy', level=0.3)
            inf_cf(i, 512, f'ablp/{mapping_dict[idx]}c7p/{i}.npy', level=0.7)
            inf_cf(i, 512, f'ablp/{mapping_dict[idx]}c9p/{i}.npy', level=0.9)

def interpo_gly(font1, font2, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(5):
        inf_cf_mid(font1, font2, 512, i/4, os.path.join(output_folder, f'{i}.npy'))

def interpo_gly_all(font1, font2):
    output_folder = f'demo/{font1}_{font2}'
    for alpha in range(52):
        model.set_glyph_idx(alpha)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for i in range(5):
            inf_cf_mid(font1, font2, 512, i/4, os.path.join(output_folder, f'{alpha}_{i}.npy'))
        
model.set_glyph_idx(0)
with torch.no_grad():
    render_whole()
    
    
