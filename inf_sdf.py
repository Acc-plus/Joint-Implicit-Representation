from models.MSDF import MultiSDF, MultiSDFSimple
from models.MCF import MultiCF
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

ep = ''
if (epoch is not None):
    ep = f' ep{epoch}'

# model = MultiSDF(num_instance, len(data_paths), **configs['model']['SignedDistanceField'])
model = MultiSDFSimple(num_instance, len(data_paths), **configs['model']['SignedDistanceField'])
assert model_toload is not None
model.load_state_dict(torch.load(f'results/{model_toload}/model{ep}.pth'))

model = model.to(device)

output_dir = os.path.join('results', model_toload)

def inference_between(c, x, y, num_frames, suf = None):
    return model.inference_multi(c, x, y, num_frames, suf)

def inference_1(c, x, suf = None):
    return model.inference(c, x, suf)

def inference_mid(c, x, y, coe):
    return model.inference_mid(c, x, y, coe)

def inf_sdf_sample(sample_id):
    print(f'generate the font sampling')

    mean_latent = model.latent_code.weight.mean(dim=0)
    std_latent = model.latent_code.weight.std(dim=0)
    emb_sample = torch.normal(mean=mean_latent, std=std_latent).to(device).unsqueeze(0)
    # emb_sample = model.latent_code.weight[0].unsqueeze(0)
    # import pdb; pdb.set_trace()

    im = np.zeros((1024, 1024), dtype=np.uint8)
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = model.inference_emb(coords, emb_sample)
        sdf_output = (sdf_output).view(-1).cpu().numpy()
        im[i, sdf_output > 0] = 255
    
    cv2.imwrite(f'results/{model_toload}/{png_save}sample{sample_id}.png', im)
    return im

resol = 512
resolution_mat = torch.FloatTensor([[[i, j] for j in range(resol)] for i in range(resol)]).to(device) / resol

def inf_sdf(x, resolution=1024, output_file=None):
    if output_file is None:
        output_file = f'results/{model_toload}/{png_save}{x}.png'
        print(f'generate the font id {x}')
    im = np.zeros((resolution, resolution), dtype=np.uint8)
    coords = resolution_mat
    sdf_output = inference_1(coords, x, None)
    sdf_output = (sdf_output).view(resolution, resolution).cpu().numpy()
    im[sdf_output > -0.000] = 255
    cv2.imwrite(output_file, im)
    return im

def inf_sdfs(x, y, num_frames=20):
    print(f'generate the font id {x}-{y}')
    ims = [np.zeros((1024, 1024), dtype=np.uint8) for _ in range(num_frames)]
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_between(coords, x, y, num_frames, None)
        sdf_output = (sdf_output).view(num_frames, -1).cpu().numpy()
        for j in range(num_frames):
            ims[j][i, sdf_output[j] > 0] = 255

    imageio.mimsave(f'results/{model_toload}/{gif_save}{x}_{y}S.gif', ims, 'GIF', duration=0.2)
    return ims

def inf_mid(x, y, resolution, coe = 0.5, output_file = None):
    if output_file is None:
        output_file = f'results/{model_toload}/{png_save}{x}_{y}_{coe}.png'
        print(f'generate the font mid {x}-{y}-{coe}')
    im = np.zeros((resolution, resolution), dtype=np.uint8)
    coords = resolution_mat
    sdf_output = inference_mid(coords, x, y, coe)
    sdf_output = (sdf_output).view(resolution, resolution).cpu().numpy()
    im[sdf_output > 0] = 255
    cv2.imwrite(output_file, im)
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

rev_gm = dict()
for gl in glyph_map.keys():
    rev_gm[glyph_map[gl]] = gl



def render_whole():
    for idx in range(52):
        if not os.path.exists(f'renders/{rev_gm[idx]}'):
            os.makedirs(f'renders/{rev_gm[idx]}')
        model.set_glyph_idx(idx)
        print(f'render {rev_gm[idx]}')
        for i in tqdm(range(1425)):
            inf_sdf(i, 512, f'renders/{rev_gm[idx]}/{i}.png')
        # break

def render_simple():
    for idx in range(52):
        if not os.path.exists(f'simple/{rev_gm[idx]}'):
            os.makedirs(f'simple/{rev_gm[idx]}')
        model.set_glyph_idx(idx)
        print(f'render {rev_gm[idx]}')
        for i in tqdm(range(1425)):
            inf_sdf(i, 512, f'simple/{rev_gm[idx]}/{i}.png')

def interpo_whole():
    import numpy as np
    sel_id = np.array(list(range(1425)), dtype=np.long)
    np.random.seed(14)
    np.random.shuffle(sel_id)
    for idx in range(52):
        if not os.path.exists(f'interpolation/{rev_gm[idx]}'):
            os.makedirs(f'interpolation/{rev_gm[idx]}')
        model.set_glyph_idx(idx)
        print(f'render {rev_gm[idx]}')
        for i in tqdm(range(500)):
            idx1 = sel_id[i*2]
            idx2 = sel_id[i*2+1]
            inf_mid(idx1, idx2, 512, 0.5, f'interpolation/{rev_gm[idx]}/{i}.png')
        # break
    
def interpo_gly(font1, font2, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(5):
        inf_mid(font1, font2, 512, i/4, os.path.join(output_folder, f'{i}.png'))
    
def interpo_gly_all(font1, font2):
    output_folder = f'demo/{font1}_{font2}'
    for alpha in range(52):
        model.set_glyph_idx(alpha)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for i in range(5):
            inf_mid(font1, font2, 512, i/4, os.path.join(output_folder, f'{alpha}_{i}.png'))

model.set_glyph_idx(0)
with torch.no_grad():
    render_whole()



    
    
