from models.DoubleIFGAN import DoubleImplicitField
from models.CornerField import CornerField
from models.SignedDistantField import SignedDistanceField
from models.OccupancyField import OccupancyField
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
data_path = configs['data']['path']
corner_path = os.path.join(data_path, 'corners')
gif_save = configs['log_info']['gif_save']
png_save = configs['log_info']['png_save']
epoch = configs['log_info']['epoch']

ep = ''
if (epoch is not None):
    ep = f' ep{epoch}'
# fontloader = CornerLoader(num_instance, data_path)

# model = TemplateOnly(coords_dim=2)
# model = ImplicitCornerField(num_instance)
# model = DoubleImplicitField(num_instance)
# model = CornerField(num_instance, activation='sine')
# model = OccupancyField(num_instance)
# model = SignedDistanceField(num_instance)
model = CornerField(num_instance)
# model = ImplicitField(num_instance, hyper_hidden_layers=2, num_hidden_layer=4)
assert model_toload is not None
model.load_state_dict(torch.load(f'results/{model_toload}/model{ep}.pth'))

model = model.to(device)
# model = nn.DataParallel(model, device_ids=dpid)

output_dir = os.path.join('results', model_toload)
# id_idx = torch.LongTensor(fontloader.font_idx).to(device)
# print(id_idx)

def inference_between(c, x, y, num_frames, suf = None):
    return model.inference_multi(c, x, y, num_frames, suf)

def inference_1(c, x, suf = None):
    return model.inference(c, x, suf)

def inference_mid(c, x, y, coe, mode = None):
    return model.inference_mid(c, x, y, coe, mode)

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

def inf_sdf(x):
    print(f'generate the font id {x}')
    im = np.zeros((1024, 1024), dtype=np.uint8)
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_1(coords, x, 'sdf')
        sdf_output = (sdf_output).view(-1).cpu().numpy()
        im[i, sdf_output > 0] = 255
    
    cv2.imwrite(f'results/{model_toload}/{png_save}{x}.png', im)
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
    im = np.zeros((1024, 1024), dtype=np.uint8)
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_mid(coords, x, y, coe, 'sdf')
        sdf_output = (sdf_output).view(-1).cpu().numpy()
        im[i, sdf_output > 0] = 255
    
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

def inf_img(x):
    print(f'generate the font id {x}')
    im = np.zeros((1024, 1024), dtype=np.uint8)
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_1(coords, x)
        sdf_output *= 255.
        sdf_output[sdf_output > 255] = 255
        sdf_output = (sdf_output).view(-1).cpu().numpy().astype(np.uint8)
        im[i, :] = sdf_output
    
    imheat = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    corner = np.load(os.path.join(corner_path, f'{x}corner.npy'))
    for cn in corner:
        cn = (cn*1024).astype(np.long)
        cv2.circle(imheat, cn, 3, [255, 255, 255], 3)
    cv2.imwrite(f'results/{model_toload}/{png_save}{x}.png', im)
    cv2.imwrite(f'results/{model_toload}/{png_save}{x}H.png', imheat)
    return im

def inf_imgs(x, y, num_frames=20):
    print(f'generate the font id {x}-{y}')
    ims = [np.zeros((1024, 1024), dtype=np.uint8) for _ in range(num_frames)]
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_between(coords, x, y, num_frames)
        sdf_output *= 255.
        sdf_output[sdf_output > 255] = 255
        sdf_output = (sdf_output).view(num_frames, -1).cpu().numpy().astype(np.uint8)
        for j in range(num_frames):
            ims[j][i, :] = sdf_output[j]
    
    imheats = [cv2.applyColorMap(ims[i], cv2.COLORMAP_JET) for i in range(num_frames)]
    imageio.mimsave(f'results/{model_toload}/{gif_save}{x}_{y}.gif', imheats, 'GIF', duration=0.2)
    return ims

def inf_maxima(x):
    print(f'generate the font id {x}')
    im = torch.zeros(1024, 1024, device=device)
    imnp = np.zeros((1024, 1024), dtype=np.uint8)
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_1(coords, x)
        im[:, i] = sdf_output.view(-1)
    imdif = torch.zeros(8, 1024, 1024, device=device)
    imdif[:] = im
    imdif[0, 1:1023, 1:1023] -= im[:1022, :1022]
    imdif[1, 1:1023, 1:1023] -= im[1:1023, :1022]
    imdif[2, 1:1023, 1:1023] -= im[2:1024, :1022]
    imdif[3, 1:1023, 1:1023] -= im[2:1024, 1:1023]
    imdif[4, 1:1023, 1:1023] -= im[2:1024, 2:1024]
    imdif[5, 1:1023, 1:1023] -= im[1:1023, 2:1024]
    imdif[6, 1:1023, 1:1023] -= im[:1022, 2:1024]
    imdif[7, 1:1023, 1:1023] -= im[:1022, 1:1023]
    imb = imdif >= 0.
    maxima = imb[0] & imb[1] & imb[2] & imb[3] & imb[4] & imb[5] & imb[6] & imb[7] & (im > 0.5)
    maxima = maxima.cpu().numpy()
    imnp[maxima] = 255
    cv2.imwrite('results/{model_toload}/MXM.png', imnp)
    print(im.max())
    return im

def inf_corners(x):
    print(f'generate the corner id {x}')
    im = np.zeros((1024, 1024), dtype=np.uint8)
    imtorch = torch.zeros(1024, 1024, device=device)
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_1(coords, x)
        imtorch[:, i] = sdf_output.view(-1)
        sdf_output *= 255.
        sdf_output[sdf_output > 255] = 255
        sdf_output = (sdf_output).view(-1).cpu().numpy().astype(np.uint8)
        im[i, :] = sdf_output
    

    imdif = torch.zeros(8, 1024, 1024, device=device)
    imdif[:] = imtorch
    imdif[0, 1:1023, 1:1023] -= imtorch[:1022, :1022]
    imdif[1, 1:1023, 1:1023] -= imtorch[1:1023, :1022]
    imdif[2, 1:1023, 1:1023] -= imtorch[2:1024, :1022]
    imdif[3, 1:1023, 1:1023] -= imtorch[2:1024, 1:1023]
    imdif[4, 1:1023, 1:1023] -= imtorch[2:1024, 2:1024]
    imdif[5, 1:1023, 1:1023] -= imtorch[1:1023, 2:1024]
    imdif[6, 1:1023, 1:1023] -= imtorch[:1022, 2:1024]
    imdif[7, 1:1023, 1:1023] -= imtorch[:1022, 1:1023]
    imb = imdif >= 0.
    maxima = imb[0] & imb[1] & imb[2] & imb[3] & imb[4] & imb[5] & imb[6] & imb[7] & (imtorch > 0.5)
    maxima = maxima.unsqueeze(2).expand(1024, 1024, 3).cpu().numpy()
    
    output_corners = []
    for i in range(1024):
        for j in range(1024):
            if (maxima[i, j, 0]):
                output_corners.append([i / 1024, j / 1024])

    np.save(f'results/{model_toload}/{png_save}{x}c.npy', np.array(output_corners))

def inf_corners_mid(x, y, coe = 0.5):
    print(f'generate the corner id {x}-{y}-{coe}')
    im = np.zeros((1024, 1024), dtype=np.uint8)
    imtorch = torch.zeros(1024, 1024, device=device)
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_mid(coords, x, y, coe)
        imtorch[:, i] = sdf_output.view(-1)
        sdf_output *= 255.
        sdf_output[sdf_output > 255] = 255
        sdf_output = (sdf_output).view(-1).cpu().numpy().astype(np.uint8)
        im[i, :] = sdf_output
    

    imdif = torch.zeros(8, 1024, 1024, device=device)
    imdif[:] = imtorch
    imdif[0, 1:1023, 1:1023] -= imtorch[:1022, :1022]
    imdif[1, 1:1023, 1:1023] -= imtorch[1:1023, :1022]
    imdif[2, 1:1023, 1:1023] -= imtorch[2:1024, :1022]
    imdif[3, 1:1023, 1:1023] -= imtorch[2:1024, 1:1023]
    imdif[4, 1:1023, 1:1023] -= imtorch[2:1024, 2:1024]
    imdif[5, 1:1023, 1:1023] -= imtorch[1:1023, 2:1024]
    imdif[6, 1:1023, 1:1023] -= imtorch[:1022, 2:1024]
    imdif[7, 1:1023, 1:1023] -= imtorch[:1022, 1:1023]
    imb = imdif >= 0.
    maxima = imb[0] & imb[1] & imb[2] & imb[3] & imb[4] & imb[5] & imb[6] & imb[7] & (imtorch > 0.5)
    maxima = maxima.unsqueeze(2).expand(1024, 1024, 3).cpu().numpy()
    
    output_corners = []
    for i in range(1024):
        for j in range(1024):
            if (maxima[i, j, 0]):
                output_corners.append([i / 1024, j / 1024])

    np.save(f'results/{model_toload}/{png_save}{x}_{y}c.npy', np.array(output_corners))

def inf_maxima_combine(x):
    print(f'generate the font id {x}')
    im = np.zeros((1024, 1024), dtype=np.uint8)
    imtorch = torch.zeros(1024, 1024, device=device)
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_1(coords, x)
        imtorch[:, i] = sdf_output.view(-1)
        sdf_output *= 255.
        sdf_output[sdf_output > 255] = 255
        sdf_output = (sdf_output).view(-1).cpu().numpy().astype(np.uint8)
        im[i, :] = sdf_output
    

    imdif = torch.zeros(8, 1024, 1024, device=device)
    imdif[:] = imtorch
    imdif[0, 1:1023, 1:1023] -= imtorch[:1022, :1022]
    imdif[1, 1:1023, 1:1023] -= imtorch[1:1023, :1022]
    imdif[2, 1:1023, 1:1023] -= imtorch[2:1024, :1022]
    imdif[3, 1:1023, 1:1023] -= imtorch[2:1024, 1:1023]
    imdif[4, 1:1023, 1:1023] -= imtorch[2:1024, 2:1024]
    imdif[5, 1:1023, 1:1023] -= imtorch[1:1023, 2:1024]
    imdif[6, 1:1023, 1:1023] -= imtorch[:1022, 2:1024]
    imdif[7, 1:1023, 1:1023] -= imtorch[:1022, 1:1023]
    imb = imdif >= 0.
    maxima = imb[0] & imb[1] & imb[2] & imb[3] & imb[4] & imb[5] & imb[6] & imb[7] & (imtorch > 0.5)
    maxima = maxima.unsqueeze(2).expand(1024, 1024, 3).cpu().numpy()
    

    imheat = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    corner = np.load(os.path.join(corner_path, f'{x}corner.npy'))
    for cn in corner:
        cn = (cn*1024).astype(np.long)
        cv2.circle(imheat, cn, 3, [255, 255, 255], 3)
    for i in range(1024):
        for j in range(1024):
            if (maxima[i, j, 0]):
                cv2.circle(imheat, [i, j], 3, [0, 0, 0], 3)
    print(maxima.sum())
    imheat[maxima] = 0
    
    cv2.imwrite(f'results/{model_toload}/{png_save}{x}HM.png', imheat)
    return im

def inf_maxima_mid_combine(x, y, coe=0.5):
    print(f'generate the font id {x} - {y}')
    im = np.zeros((1024, 1024), dtype=np.uint8)
    imtorch = torch.zeros(1024, 1024, device=device)
    for i in range(1024):
        coords = torch.FloatTensor([[i, j] for j in range(1024)]).to(device) / 1024.
        sdf_output = inference_mid(coords, x, y, coe)
        imtorch[:, i] = sdf_output.view(-1)
        sdf_output *= 255.
        sdf_output[sdf_output > 255] = 255
        sdf_output = (sdf_output).view(-1).cpu().numpy().astype(np.uint8)
        im[i, :] = sdf_output
    

    imdif = torch.zeros(8, 1024, 1024, device=device)
    imdif[:] = imtorch
    imdif[0, 1:1023, 1:1023] -= imtorch[:1022, :1022]
    imdif[1, 1:1023, 1:1023] -= imtorch[1:1023, :1022]
    imdif[2, 1:1023, 1:1023] -= imtorch[2:1024, :1022]
    imdif[3, 1:1023, 1:1023] -= imtorch[2:1024, 1:1023]
    imdif[4, 1:1023, 1:1023] -= imtorch[2:1024, 2:1024]
    imdif[5, 1:1023, 1:1023] -= imtorch[1:1023, 2:1024]
    imdif[6, 1:1023, 1:1023] -= imtorch[:1022, 2:1024]
    imdif[7, 1:1023, 1:1023] -= imtorch[:1022, 1:1023]
    imb = imdif >= 0.
    maxima = imb[0] & imb[1] & imb[2] & imb[3] & imb[4] & imb[5] & imb[6] & imb[7] & (imtorch > 0.5)
    maxima = maxima.unsqueeze(2).expand(1024, 1024, 3).cpu().numpy()
    

    imheat = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    for i in range(1024):
        for j in range(1024):
            if (maxima[i, j, 0]):
                cv2.circle(imheat, [i, j], 3, [0, 0, 0], 3)
    
    print(maxima.sum())
    imheat[maxima] = 0
    output_corners = []
    for i in range(1024):
        for j in range(1024):
            if (maxima[i, j, 0]):
                output_corners.append([i / 1024, j / 1024])
       
    cv2.imwrite(f'results/{model_toload}/{png_save}{x}_{y}_{coe}HM.png', imheat)
    np.save(f'results/{model_toload}/{png_save}{x}_{y}_{coe}c.npy', np.array(output_corners))
    return im

def print_corner(filename):
    
    canvas = np.zeros((1024, 1024), dtype=np.uint8)
    corners = np.load(filename)
    print(corners)
    corners[7, 0] = 0.63
    for cn in corners:
        cn = (cn*1024).astype(np.long)
        cv2.circle(canvas, cn, 3, 255, 3)

    cv2.imwrite('outc.png', canvas)
    np.save('aug4.npy', corners)

print('--rendering start--')
model.eval()

def sdf_testblock():
    inf_sdfs(339, 2359)
    inf_sdfs(2353, 1299)
    inf_sdfs(1253, 1299)
    inf_sdfs(1403, 1321)
    inf_sdf(1436)
    inf_sdf(5619)

# model.L1Regularization = True
with torch.no_grad():
    # inf_imgs(1436, 1321)
    # inf_sdfs(1436, 1321)
    for i in range(11):
        inf_maxima_mid_combine(1436, 1321, i/10)
        # inf_sdf_mid(1436, 1321, i/10)
    # for i in range(11):
        
    
    
