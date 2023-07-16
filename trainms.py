from utils.dataloader import *
from torch.utils.data import DataLoader
from models.MSDF import MultiSDF, MultiSDFSimple
from models.MCF import MultiCF, MultiCFSimple
from models.MCF_EX import MultiCFEX
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


model_toload = configs['log_info']['ckpt_load']
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

fontloader = MultiSDFLoader(num_instance, data_paths)
# model = MultiSDF(num_instance, len(data_paths), **configs['model']['SignedDistanceField'])
model = MultiSDFSimple(num_instance, len(data_paths), **configs['model']['SignedDistanceField'])


# fontloader = MultiCFLoader(num_instance, data_paths)
# model = MultiCF(num_instance, len(data_paths), **configs['model']['CornerField'])
# model = MultiCFSimple(num_instance, len(data_paths), **configs['model']['CornerField'])
# model = MultiCFEX(num_instance, len(data_paths), **configs['model']['CornerField'])
# import pdb; pdb.set_trace()
dataloader = DataLoader(fontloader, shuffle=True,batch_size=configs['training']['batch_size'], pin_memory=False, num_workers=16, drop_last = True, collate_fn=None, worker_init_fn=lambda worker_id: np.random.seed(12345 + worker_id))

if model_toload is not None:
    model.load_state_dict(torch.load(f'results/{model_toload}/model{ep}.pth'))

model = model.to(device)
model = nn.DataParallel(model, device_ids=dpid)

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

optimizers.append(torch.optim.Adam(lr=configs['training']['LR']*(configs['training']['lr_decay']**(start_ep+1)), params=model.parameters()))
lrschedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizers[0], configs['training']['lr_decay']))
# lrschedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=25, last_epoch=-1))

lendata = len(dataloader)

keyword_loss = 'sdf' if (model.module.__class__ == MultiSDF or model.module.__class__ == MultiSDFSimple) else 'corner'
suf_msg = '--training-start--'


model.train()
for epoch in range(start_ep+1, configs['training']['epochs']):
    print(suf_msg)
    pbar = tqdm(dataloader)
    totloss = 0
    for iter, inputs in enumerate(pbar):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        losses = model.forward(inputs)
        
        train_loss = 0.
        for loss_name, loss in losses.items():
            single_loss = loss.mean()
            train_loss += single_loss
        totloss += losses[keyword_loss].mean().item()
        # if iter % 100 == 0:
            # print(f'--epoch {epoch}--iter {iter}--loss {train_loss.item()}--')
        pbar.set_postfix({
            'epoch': f'{epoch}',
            'loss': f'{losses[keyword_loss].mean().item()}'
        })
        suf_msg = f'{losses.items()}'
        # if (iter == lendata - 1):
        #     print(losses.items())
        for opt in optimizers:
            opt.zero_grad()
        train_loss.backward()
        for opt in optimizers:
            opt.step()
    for sch in lrschedulers:
        sch.step()
    
    with open(f'results/{model_tosave}/loss.txt', 'a') as f:
        f.write(f'{totloss}\n')
    if (epoch % 50) == 0:
        torch.save(model.module.state_dict(), f'results/{model_tosave}/model ep{epoch}.pth')


torch.save(model.module.state_dict(), f'results/{model_tosave}/model.pth')
exit()

