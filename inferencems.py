from models.DoubleIFGAN import DoubleImplicitField
from models.CornerField import CornerField
from models.SignedDistantField import SignedDistanceField
from models.OccupancyField import OccupancyField
from models.MSDF import MultiSDF
from models.MCF import MultiCF
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

model_toload = configs['log_info']['pretrained']
num_instance = configs['data']['num_instance']
data_paths = configs['data']['paths']
gif_save = configs['log_info']['gif_save']
png_save = configs['log_info']['png_save']
epoch = configs['log_info']['epoch']

ep = ''
if (epoch is not None):
    ep = f' ep{epoch}'

model = MultiSDF(num_instance, len(data_paths), **configs['model']['SignedDistanceField'])
# model = MultiCF(num_instance, len(data_paths), **configs['model']['CornerField'])
assert model_toload is not None
model.load_state_dict(torch.load(f'results/{model_toload}/model{ep}.pth'))

model = model.to(device)

output_dir = os.path.join('results', model_toload)

def inference_between(c, x, y, num_frames, suf = None):
    return model.inference_multi(c, x, y, num_frames, suf)

def inference_1(c, x, suf = None):
    return model.inference(c, x, suf)

def inference_mid(c, x, y, coe, mode = None):
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
    # for i in range(resolution):
    #     coords = torch.FloatTensor([[i, j] for j in range(resolution)]).to(device) / resolution
    #     sdf_output = inference_1(coords, x, 'sdf')
    #     sdf_output = (sdf_output).view(-1).cpu().numpy()
    #     im[i, sdf_output > 0] = 255
    coords = resolution_mat
    sdf_output = inference_1(coords, x, 'sdf')
    sdf_output = (sdf_output).view(resolution, resolution).cpu().numpy()
    im[sdf_output > 0] = 255
    cv2.imwrite(output_file, im)
    return im

def inf_cf(x, resolution=1024, output_file=None):
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
    # cv2.imwrite(output_file, imheat)

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
       
    # cv2.imwrite(f'results/{model_toload}/{png_save}{x}H.png', imheat)
    np.save(output_file, np.array(maxima))
    return im

def inf_occ(x):
    print(f'generate the font id {x}')
    im = np.zeros((1024, 1024), dtype=np.uint8)
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_1(coords, x, 'sdf')
        sdf_output = (sdf_output).view(-1).cpu().numpy()
        # import pdb; pdb.set_trace()
        im[i, sdf_output > 0.5] = 255
        im[i, sdf_output <= 0.5] = 0
    
    cv2.imwrite(f'results/{model_toload}/{png_save}{x}.png', im)
    return im

def inf_sdfs(x, y, num_frames=20):
    print(f'generate the font id {x}-{y}')
    ims = [np.zeros((1024, 1024), dtype=np.uint8) for _ in range(num_frames)]
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_between(coords, x, y, num_frames, 'sdf')
        sdf_output = (sdf_output).view(num_frames, -1).cpu().numpy()
        for j in range(num_frames):
            ims[j][i, sdf_output[j] > 0] = 255

    imageio.mimsave(f'results/{model_toload}/{gif_save}{x}_{y}S.gif', ims, 'GIF', duration=0.2)
    return ims

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

def inf_sdfs_sample(sample_id, num_frames=20):
    print(f'generate the font id interpolation sample')
    mean_latent = model.latent_code.weight.mean(dim=0)
    std_latent = model.latent_code.weight.std(dim=0)
    # min_latent, _ = model.latent_code.weight.min(dim=0)
    # max_latent, _ = model.latent_code.weight.max(dim=0)
    # margin = max_latent - min_latent
    emb_sample1 = torch.normal(mean=mean_latent, std=std_latent).to(device)
    emb_sample2 = torch.normal(mean=mean_latent, std=std_latent).to(device)
    # emb_sample1 = (torch.rand(model.latent_dim).to(device)*margin+min_latent)
    # emb_sample2 = (torch.rand(model.latent_dim).to(device)*margin+min_latent)
    ims = [np.zeros((1024, 1024), dtype=np.uint8) for _ in range(num_frames)]
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = model.inference_multi_emb(coords, emb_sample1, emb_sample2, num_frames)
        sdf_output = (sdf_output).view(num_frames, -1).cpu().numpy()
        for j in range(num_frames):
            ims[j][i, sdf_output[j] > 0] = 255

    imageio.mimsave(f'results/{model_toload}/{gif_save}sample{sample_id}S.gif', ims, 'GIF', duration=0.2)
    return ims

def inf_sdf_mid(x, y, coe = 0.5):
    print(f'generate the font mid {x}-{y}-{coe}')
    sdf_output = inference_mid(resolution_mat, x, y, coe, 'sdf')
    im = np.zeros((512, 512), dtype=np.uint8)
    sdf_output = sdf_output.view(512, 512).cpu().numpy()
    im[sdf_output > 0] = 255
    cv2.imwrite(f'results/{model_toload}/{png_save}{x}_{y}_{coe}.png', im)
    return im


def inf_diff(x, num_frames):
    print(f'generate the font id {x}')
    ims = [np.zeros((1024, 1024), dtype=np.uint8) for _ in range(num_frames)]
    model.generate_lchypos(x, num_frames)
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = model.generate_diff(coords, num_frames)
        sdf_output = (sdf_output).view(num_frames, -1).cpu().numpy()
        for j in range(num_frames):
            ims[j][i, sdf_output[j] > 0] = 255

    imageio.mimsave(f'results/{model_toload}/{gif_save}{x}DiffS.gif', ims, 'GIF', duration=0.2)
    return ims



print('--rendering start--')
model.eval()

mapping_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
}



def inf_sdf_mid_output(x, y, coe = 0.5, output_file=None):
    print(f'generate the font mid {x}-{y}-{coe}')
    sdf_output = inference_mid(resolution_mat, x, y, coe, 'sdf')
    im = np.zeros((512, 512), dtype=np.uint8)
    sdf_output = sdf_output.view(512, 512).cpu().numpy()
    im[sdf_output > 0] = 255
    cv2.imwrite(output_file, im)
    return im
model.set_glyph_idx(0)
with torch.no_grad():
    # inf_sdf_mid(89, 964, 0.3)
    # inf_sdf_mid(1132, 1028, 0.025*17)
    for i in range(26):
        model.set_glyph_idx(i)
        inf_sdf_mid_output(220, 1295, 0.025*16, f'rebuttal/{i}.png')
    # inf_cfs(2, 3)
    # idx = 0
    # model.set_glyph_idx(idx)
    # for i in tqdm(range(1425)):
    #     # inf_sdf(i, 512, f'evaluation/{mapping_dict[idx]}/{i}.png')
    #     inf_cf(i, 512, f'evaluation/{mapping_dict[idx]}c/{i}.npy')
    # inf_cf(12, 512)
    # inf_sdf(12, 512)
    # inf_sdfs(279, 279)
    # inf_sdfs(339, 2359)
    # inf_sdfs(2353, 1299)
    # inf_sdfs(1253, 1299)
    # inf_sdfs(1403, 1321)
    # inf_sdfs(1436, 1321)
    # inf_sdfs(29, 31)
    # inf_sdfs(29, 59)
    # inf_sdfs(110, 111)
    # inf_sdfs(133, 134)
    # inf_sdfs(2044, 2045)
    # inf_cfs(1187, 1189)

    
    
