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

def main():
    setup_seed(configs['seed'])
    model_toload = configs['log_info']['ckpt_load']
    model_tosave = configs['log_info']['ckpt_save']
    num_instance = configs['data']['num_instance']
    data_paths = configs['data']['paths']
    epoch = configs['log_info']['epoch']
    save_freq = configs['training']['save_freq']
    model_class = eval(configs['model']['type'])
    data_class = eval(configs['data']['type'])

    ep = ''
    if (epoch is not None):
        ep = f' ep{epoch}'

    assert not os.path.exists(os.path.join('results', model_tosave))
    os.makedirs(os.path.join('results', model_tosave), exist_ok = True)

    fontloader = data_class(num_instance, data_paths)
    model = model_class(num_instance, len(data_paths), **configs['model']['params'])
    dataloader = DataLoader(fontloader, shuffle=True,batch_size=configs['training']['batch_size'], pin_memory=False, num_workers=16, drop_last = True, collate_fn=None, worker_init_fn=lambda worker_id: np.random.seed(12345 + worker_id))

    if model_toload is not None:
        model.load_state_dict(torch.load(f'results/{model_toload}/model{ep}.pth'))

    model = model.to(device)
    model = nn.DataParallel(model, device_ids=dpid)
        
    start_ep = -1 if epoch is None else epoch

    optimizers = []
    # LR scheduler only support update without loss info
    lrschedulers = []

    optimizers.append(torch.optim.Adam(lr=configs['training']['LR']*(configs['training']['lr_decay']**(start_ep+1)), params=model.parameters()))
    lrschedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizers[0], configs['training']['lr_decay']))

    lendata = len(dataloader)
    keyword_loss = 'sdf' if (model.module.__class__ == MultiSDF or model.module.__class__ == MultiSDFSimple) else 'corner'
    suf_msg = '--training-start--'
    print(suf_msg)

    model.train()
    pbar = tqdm(range(start_ep+1, configs['training']['epochs']))
    for epoch in pbar:
        
        totloss = {}
        for iter, inputs in enumerate(dataloader):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            losses = model.forward(inputs)
            
            train_loss = 0.
            if iter == 0:
                for key in losses:
                    totloss[key] = 0.
            for key, loss in losses.items():
                single_loss = loss.mean()
                train_loss += single_loss
                totloss[key] += single_loss.item()
            pbar.set_postfix({
                'epoch': f'{epoch}',
                'iter': f'{iter}/{lendata}',
                'loss': f'{losses[keyword_loss].mean().item():.2f}'
            })
            for opt in optimizers:
                opt.zero_grad()
            train_loss.backward()
            for opt in optimizers:
                opt.step()
        for sch in lrschedulers:
            sch.step()
        
        with open(f'results/{model_tosave}/loss.txt', 'a') as f:
            f.write(f'epoch{epoch}:{totloss}\n')
        if ((epoch+1) % save_freq) == 0:
            torch.save(model.module.state_dict(), f'results/{model_tosave}/model ep{epoch}.pth')

    torch.save(model.module.state_dict(), f'results/{model_tosave}/model.pth')

if __name__ == '__main__':
    main()