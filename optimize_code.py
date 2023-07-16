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
# fit_idx = 260
# fit_idx = 203 - D
# fit idx = 169 - A
# fit_idx = 2 - C
# fit_idx = 20
# fit_idx = 19
# fit_idx = 1301
# 748 -> 754
# fit_idx = 612
# fit_idx = 710
fit_idx = 1025
init_idx = fit_idx
singleImageLoader = SingleImageSDF(data_paths[glyph_idx], glyph_idx, fit_idx)
# import pdb; pdb.set_trace()
data_paths = ['/mnt/data1/cjh/Datasets/A_Train0']
num_instance = 2000
fontloader = MultiSDFLoader(num_instance, data_paths[:1], minibatch=1)
import pdb; pdb.set_trace()
model = MultiSDF(1500, 52, **configs['model']['SignedDistanceField'])

dataloader = DataLoader(fontloader, shuffle=True,batch_size=configs['training']['batch_size'], pin_memory=False, num_workers=16, drop_last = True, collate_fn=None, worker_init_fn=lambda worker_id: np.random.seed(12345 + worker_id))

if model_toload is not None:
    model.load_state_dict(torch.load(f'results/{model_toload}/model{ep}.pth', map_location=torch.device('cpu')))

model = model.to(device)
# model = nn.DataParallel(model, device_ids=dpid)

# def param_filter(model, special_module_names = []):
#     special_module_names.append('')
#     parameter_groups = [[] for _ in special_module_names]
#     for name, param in model.named_parameters():
#         for i, sp_name in enumerate(special_module_names):
#             if sp_name in name:
#                 parameter_groups[i].append(param)
#                 break
#     return parameter_groups
    
start_ep = -1 if epoch is None else epoch
    
# parameter_groups = param_filter(model)

optimizers = []
# LR scheduler only support update without loss info
lrschedulers = []

code_font = torch.zeros_like(model.glyph_code.weight[0], device=device)
# nn.init.normal_(code_font, mean=0, std=0.0001)
code_font.requires_grad = True
optimizers.append(torch.optim.Adam(lr=0.0005, params=[code_font]))
lrschedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizers[0], 0.99992))
# lrschedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=25, last_epoch=-1))

lendata = len(dataloader)

keyword_loss = 'sdf' 
suf_msg = '--training-start--'


# fit_idx = 138

min_loss = 1000.

# mt = cv2.imread(f'../Datasets/A_Train0/{fit_idx}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
# for i in tqdm(range(1425)):
#     timg = cv2.imread(f'../dvfdata/A_Test0/{i}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
#     ls = np.abs(timg - mt).mean()
#     if ls < min_loss:
#         print(ls)
#         min_loss = ls
#         init_idx = i
# print(init_idx)
# with torch.no_grad():
#     code_font[:] = model.latent_code.weight[init_idx]

fontloader.SDFLoaders[0].nSample_edge = 5000
# fontloader.SDFLoaders[0].nSample_edge = 0
# fontloader.SDFLoaders[0].nSample_free = 2000
fontloader.SDFLoaders[0].nSample_free = 5000
# import pdb; pdb.set_trace()
model.train()
pbar = tqdm(range(20000))
print(suf_msg)
f_data = fontloader[fit_idx]
# f_data = singleImageLoader[0]
totloss = 0
inputs = {key: torch.from_numpy(np.array([value])).to(device) for key, value in f_data.items()}
# inputs['sdf'] /= 1.4
inputs['font'] = code_font
for epoch in enumerate(pbar):
    losses = model.forward_embedding(inputs)
    # import pdb; pdb.set_trace()
    train_loss = 0.
    for loss_name, loss in losses.items():
        single_loss = loss.mean()
        train_loss += single_loss
    # train_loss += losses['sdf'].mean()
    totloss += losses[keyword_loss].mean().item()
    pbar.set_postfix({
        'epoch': f'{epoch}',
        'loss': f'{losses[keyword_loss].mean().item()}'
    })
    suf_msg = f'{losses.items()}'
    for opt in optimizers:
        opt.zero_grad()
    train_loss.backward()
    for opt in optimizers:
        opt.step()
    for sch in lrschedulers:
        sch.step()
    # import pdb; pdb.set_trace()
model.eval()

png_save = configs['log_info']['png_save']
resol = 512
resolution_mat = torch.FloatTensor([[[i, j] for j in range(resol)] for i in range(resol)]).to(device) / resol
# import pdb; pdb.set_trace()
def inference_1(c, x, emb = None):
    return model.inference(c, x, emb)
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


with torch.no_grad():
    for i in range(26):
        model.set_glyph_idx(i)
        inf_sdf(0, 512, f'{i}zzzz.png')
    import pdb; pdb.set_trace()
    model.set_glyph_idx(35)
    inf_sdf(0, 512)

