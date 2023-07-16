from models.GridPatchSampler import GridPatchSampler
from utils.dataloader import *
from models.disc import ConsistencyDiscriminator
from torch.utils.data import DataLoader
from models.CornerField import CornerField
from models.SignedDistantField import SignedDistanceField
from models.OccupancyField import OccupancyField
import torch
import torch.nn as nn
import random
import numpy as np
from globalconfig import *
from datetime import *
import torchvision

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
data_path = configs['data']['path']
data_expath = configs['data']['extra_path']
epoch = configs['log_info']['epoch']

ep = ''
if (epoch is not None):
    ep = f' ep{epoch}'

if not os.path.exists(os.path.join('results', model_tosave)):
    os.makedirs(os.path.join('results', model_tosave))

fontloader = CornerLoader(num_instance, data_path, mini_batch=8, extra_dir=data_expath)
# fontloader = OccupancyLoader(num_instance, data_path)
# fontloader = SdfLoader(num_instance, data_path)
# fontloader = FpicLoader(num_instance, data_path)
# fontloader = CombinedLoader(num_instance, data_path)
# fontloader = ConvLoader(num_instance, data_path)


dataloader = DataLoader(fontloader, shuffle=True,batch_size=configs['training']['batch_size'], pin_memory=False, num_workers=16, drop_last = True, collate_fn=None, worker_init_fn=lambda worker_id: np.random.seed(12345 + worker_id))


model = CornerField(num_instance, L1_reg=False)
# discriminator = torchvision.models.resnet18(num_classes=1)
discriminator = ConsistencyDiscriminator(1)
discriminator.load_state_dict(torch.load('discrL1.pth'))
# discriminator = torch.from_numpy(np.load('gtemplate.npy')).to(device)
PatchSampler = GridPatchSampler(
    model.latent_code, 
    model.corner_net, 
    model.hyper_corner_net,
    discriminator
)


if model_toload is not None:
    model.load_state_dict(torch.load(f'results/{model_toload}/model{ep}.pth'))



model = model.to(device)
PatchSampler = PatchSampler.to(device)
# model = nn.DataParallel(model, device_ids=dpid)

optimizers = []
# LR scheduler only support update without loss info
lrschedulers = []

optimizers.append(torch.optim.Adam(lr=configs['training']['LR'], params=model.parameters()))
lrschedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizers[0], configs['training']['lr_decay']))
# lrschedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=25, last_epoch=-1))

lendata = len(dataloader)

# with open('netarch.txt', 'w') as f:
#     f.write(model.__repr__())

model.train()
print('--training-start--')
for epoch in range(configs['training']['epochs']):
    
    totloss = 0
    for iter, inputs in enumerate(dataloader):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        losses = model.forward(inputs)
        discriminator_loss = PatchSampler(inputs)
        losses['patch'] = discriminator_loss
        train_loss = 0.
        with open(f'results/{model_tosave}/patchloss.txt', 'a') as f:
            f.write(f'{discriminator_loss.item()}\n')
        for loss_name, loss in losses.items():
            single_loss = loss.mean()
            train_loss += single_loss
        if (iter == lendata - 1):
            print(losses.items())
        totloss += train_loss.item()
        print(f'--epoch {epoch}--iter {iter}--loss {train_loss.item()}--')

        for opt in optimizers:
            opt.zero_grad()
        train_loss.backward()
        for opt in optimizers:
            opt.step()
    for sch in lrschedulers:
        sch.step()
    
    with open(f'results/{model_tosave}/loss.txt', 'a') as f:
        f.write(f'{totloss}\n')

    if (epoch+1) % 25 == 0:
        torch.save(model.state_dict(), f'results/{model_tosave}/model ep{epoch}.pth')
torch.save(model.state_dict(), f'results/{model_tosave}/model.pth')
exit()

