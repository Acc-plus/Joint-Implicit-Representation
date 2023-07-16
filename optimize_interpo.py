from models.DoubleIFGAN import DoubleImplicitField
from utils.dataloader import *
from torch.utils.data import DataLoader
from models.CornerField import CornerField
from models.SignedDistantField import SignedDistanceField
from models.MSDF import MultiSDF
from models.MCF import MultiCF
from models.MCF_EX import MultiCFEX
from models.OccupancyField import OccupancyField
import torch
import torch.nn as nn
import random
import numpy as np
from globalconfig import *
from datetime import *
from tqdm import tqdm

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
setup_seed(configs['seed'])


model_toload = configs['log_info']['pretrained']
gif_tosave = configs['log_info']['ckpt_load']
model_tosave = configs['log_info']['ckpt_save']
num_instance = configs['data']['num_instance']
data_paths = configs['data']['paths']
data_expath = configs['data']['extra_path']
epoch = configs['log_info']['epoch']

ep = ''
if (epoch is not None):
    ep = f' ep{epoch}'

if not os.path.exists(os.path.join('results', model_tosave)):
    os.makedirs(os.path.join('results', model_tosave))

glyph_idx = 0
# fit_idx = 622
fit_idx = 628
init_idx = fit_idx

model = MultiSDF(1500, 52, **configs['model']['SignedDistanceField'])

if model_toload is not None:
    model.load_state_dict(torch.load(f'results/{model_toload}/model{ep}.pth'))

model = model.to(device)
# model = nn.DataParallel(model, device_ids=dpid)

start_ep = -1 if epoch is None else epoch
    
# parameter_groups = param_filter(model)

optimizers = []
# LR scheduler only support update without loss info
lrschedulers = []

code_font = torch.zeros_like(model.glyph_code.weight[0], device=device)
# nn.init.normal_(code_font, mean=0, std=0.0001)
code_font.requires_grad = True
optimizers.append(torch.optim.Adam(lr=0.00005, params=[code_font]))
lrschedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizers[0], 0.999))
# lrschedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=25, last_epoch=-1))


keyword_loss = 'sdf' 
suf_msg = '--training-start--'


# fit_idx = 138

min_loss = 1000.

imgs = np.zeros((1425, 512, 512)).astype(np.float32)
errors = np.zeros(1425)
mt = cv2.imread(f'../Datasets/A_Train0/{fit_idx}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
mt = cv2.resize(mt, (512, 512))
mt_dev = torch.from_numpy(mt).to(device) / 255
for i in tqdm(range(1425)):
    timg = cv2.imread(f'../dvfdata/A_Test0/{i}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    timg = cv2.resize(timg, (512, 512))
    ls = np.abs(timg - mt).mean()
    errors[i] = ls
    if ls < min_loss:
        # print(ls)
        min_loss = ls
        init_idx = i
        print(i)
    imgs[i] = timg
print(init_idx)
init_idx = 220
with torch.no_grad():
    code_font[:] = model.latent_code.weight[init_idx]

interpo_idx = 0

model.eval()
pbar = tqdm(range(1425))
print(suf_msg)

        


png_save = configs['log_info']['png_save']
resol = 512
resolution_mat = torch.FloatTensor([[[i, j] for j in range(resol)] for i in range(resol)]).to(device) / resol
# import pdb; pdb.set_trace()
def inference_1(c, x, emb = None):
    return model.inference(c, x, emb)

def inf_sdfnp(x, resolution=512):
    if output_file is None:
        output_file = f'results/{model_toload}/{png_save}{x}.png'
        print(f'generate the font id {x}')
    im = np.zeros((resolution, resolution), dtype=np.uint8)
    coords = resolution_mat
    sdf_output = inference_1(coords, x, code_font)
    sdf_output = (sdf_output).view(resolution, resolution).cpu().numpy()
    im[sdf_output > -0.000] = 255
    return im

def inf_sdf(x, resolution=1024, output_file=None):
    if output_file is None:
        output_file = f'results/{model_toload}/{png_save}{x}.png'
        print(f'generate the font id {x}')
    im = np.zeros((resolution, resolution), dtype=np.uint8)
    coords = resolution_mat
    sdf_output = inference_1(coords, x, code_font)
    sdf_output = (sdf_output).view(resolution, resolution).cpu().numpy()
    im[sdf_output > -0.000] = 255
    cv2.imwrite(output_file, im)
    return im

code_init = model.latent_code.weight[init_idx]
tmp_minloss = 1000.
tmp_code = 0
tmp_coe = 0
with torch.no_grad():
    for idx in enumerate(pbar):
        # searching error bar
        if errors[idx[0]] > 40:
            continue
        for icoe in range(20):
            code_idx = model.latent_code.weight[idx[0]]
            coords = resolution_mat
            emb = code_init * (1-icoe*0.025) + code_idx * (icoe * 0.025)
            outputs = model.infernce_embedding({
                'coords': coords,
                'font': emb,
                'glyph_idx': torch.LongTensor([glyph_idx]).to(device)
            })
            outputs = outputs.view(512, 512)
            im_outputs = torch.zeros((512, 512), device=device)
            im_outputs[outputs>0] = 1
            loss = torch.abs(mt_dev - im_outputs).mean()
            if loss < tmp_minloss:
                tmp_minloss = loss
                tmp_code = idx
                tmp_coe = icoe
                
            pbar.set_postfix({
                'idx': f'{idx}',
            })

print(tmp_code, tmp_coe)
# with torch.no_grad():
#     model.set_glyph_idx(0)
#     inf_sdf(0, 512)
#     import pdb; pdb.set_trace()
#     model.set_glyph_idx(35)
#     inf_sdf(0, 512)

