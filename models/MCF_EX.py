import torch
from torch import nn
import models.modules as modules
from models.meta_modules import HyperNetwork
from torchmeta.modules import (MetaModule, MetaSequential)
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F
from globalconfig import *
import random

class MultiCFEX(nn.Module):
    def __init__(self, 
    num_instances,
    num_glyph,
    latent_dim=128, 
    hyper_hidden_layers=1,
    hyper_hidden_features=256,
    hidden_num=128, 
    num_hidden_layer=3,
    activation='sine',
    glyph_dim=32, **kwargs):
        super().__init__()
        hyper_hidden_layers = 2
        num_hidden_layer = 1

        self.sdflow_feature = 1 
        self.latent_dim = latent_dim
        self.glyph_dim = glyph_dim
        self.num_instances = num_instances
        self.latent_code = nn.Embedding(num_instances, self.latent_dim)
        self.glyph_code = nn.Embedding(num_glyph, self.glyph_dim)
        nn.init.normal_(self.latent_code.weight, mean=0, std=0.01)
        nn.init.normal_(self.glyph_code.weight, mean=0, std=0.01)
        self.extra_embedding = False
        if self.extra_embedding:
            self.extra_code = nn.Embedding(num_instances, self.latent_dim)
            nn.init.normal_(self.extra_code.weight, mean=0, std=0.01)
            self.register_buffer('extra_coe', torch.rand((num_instances, ))*0.4 + 0.3)
            
        self.corner_net1=modules.SingleBVPNet(type=activation,mode='mlp', 
                                             hidden_features=hidden_num, 
                                             num_hidden_layers=num_hidden_layer, 
                                             in_features=2,
                                             out_features=hidden_num)
        self.hyper_corner_net1 = HyperNetwork(hyper_in_features=self.latent_dim, 
                                             hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, 
                                             hypo_module=self.corner_net1)
        self.hyper_glyph_net1 = HyperNetwork(hyper_in_features=self.glyph_dim, 
                                             hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, 
                                             hypo_module=self.corner_net1)

        self.corner_net2 = modules.SingleBVPNet(type=activation,mode='mlp', 
                                             hidden_features=hidden_num, 
                                             num_hidden_layers=num_hidden_layer, 
                                             in_features=hidden_num,
                                             out_features=1+self.sdflow_feature)
        self.hyper_corner_net2 = HyperNetwork(hyper_in_features=self.latent_dim, 
                                             hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, 
                                             hypo_module=self.corner_net2)
        self.hyper_glyph_net2 = HyperNetwork(hyper_in_features=self.glyph_dim, 
                                             hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, 
                                             hypo_module=self.corner_net2)
    def set_glyph_idx(self, idx):
        self.glyph_idx = torch.LongTensor([idx]).to(device)

    def inference(self, coords, instance_idx, emb_mode = None):
        coords = coords.unsqueeze(0)
        instance_idx = torch.LongTensor([instance_idx]).to(device)
        embedding = self.latent_code(instance_idx)
        emebdding_glyph = self.glyph_code(self.glyph_idx)
        hypo_cf1 = self.hyper_corner_net1(embedding)
        hypo_glyph1 = self.hyper_glyph_net1(emebdding_glyph.unsqueeze(0))
        hypo_cf2 = self.hyper_corner_net2(embedding)
        hypo_glyph2 = self.hyper_glyph_net2(emebdding_glyph.unsqueeze(0))
        hypos1 = OrderedDict()
        hypos2 = OrderedDict()
        for key in hypo_cf1.keys():
            hypos1[key] = hypo_cf1[key] + hypo_glyph1[key]
        for key in hypo_cf2.keys():
            hypos2[key] = hypo_cf2[key] + hypo_glyph2[key]
        output_mids = self.corner_net1({'coords': coords}, params=hypos1)
        output_cf = self.corner_net2({'coords': output_mids['model_out']}, params=hypos2)
        return torch.sigmoid(output_cf['model_out'][:, :, 0])

    def inference_multi(self, coords, instance_idx1, instance_idx2, num_frames, emb_mode = None):
        coords = coords.unsqueeze(0).expand(num_frames, -1, 2)
        instance_idx1 = torch.LongTensor([instance_idx1]).to(device)
        instance_idx2 = torch.LongTensor([instance_idx2]).to(device)
        embedding1 = self.latent_code(instance_idx1)
        embedding2 = self.latent_code(instance_idx2)
        emebdding_glyph = self.glyph_code(self.glyph_idx)
        
        c = torch.arange(0,1,1/num_frames).to(device)
        c[num_frames - 1] = 1.
        cr = 1. - c
        embedding = torch.zeros(num_frames, self.latent_dim).to(device)
        for i in range(num_frames):
            embedding[i] = c[i] * embedding2 + cr[i] * embedding1

        hypo_cf1 = self.hyper_corner_net1(embedding)
        hypo_glyph1 = self.hyper_glyph_net1(emebdding_glyph.expand(num_frames, self.glyph_dim))
        hypo_cf2 = self.hyper_corner_net2(embedding)
        hypo_glyph2 = self.hyper_glyph_net2(emebdding_glyph.expand(num_frames, self.glyph_dim))
        hypos1 = OrderedDict()
        hypos2 = OrderedDict()
        for key in hypo_cf1.keys():
            hypos1[key] = hypo_cf1[key] + hypo_glyph1[key]
        for key in hypo_cf2.keys():
            hypos2[key] = hypo_cf2[key] + hypo_glyph2[key]
        output_mids = self.corner_net1({'coords': coords}, params=hypos1)
        output_cf = self.corner_net2({'coords': output_mids['model_out']}, params=hypos2)
        return torch.sigmoid(output_cf['model_out'][:, :, 0])

    def inference_mid(self, coords, instance_idx1, instance_idx2, coe, **kwargs):
        coords = coords.unsqueeze(0)
        instance_idx1 = torch.LongTensor([instance_idx1]).to(device)
        instance_idx2 = torch.LongTensor([instance_idx2]).to(device)

        embedding1 = self.latent_code(instance_idx1)
        embedding2 = self.latent_code(instance_idx2)
        emebdding_glyph = self.glyph_code(self.glyph_idx)
        embedding = embedding1 * (1 - coe) + embedding2 * coe
        
        hypo_cf1 = self.hyper_corner_net1(embedding)
        hypo_glyph1 = self.hyper_glyph_net1(emebdding_glyph.unsqueeze(0))
        hypo_cf2 = self.hyper_corner_net2(embedding)
        hypo_glyph2 = self.hyper_glyph_net2(emebdding_glyph.unsqueeze(0))
        hypos1 = OrderedDict()
        hypos2 = OrderedDict()
        for key in hypo_cf1.keys():
            hypos1[key] = hypo_cf1[key] + hypo_glyph1[key]
        for key in hypo_cf2.keys():
            hypos2[key] = hypo_cf2[key] + hypo_glyph2[key]
        output_mids = self.corner_net1({'coords': coords}, params=hypos1)
        output_cf = self.corner_net2({'coords': output_mids['model_out']}, params=hypos2)
        return torch.sigmoid(output_cf['model_out'][:, :, 0])
    
    def forward(self, model_input):
    
        instance_idx = model_input['instance_idx']
        coords_corner =  {'coords': model_input['coords']}
        glyph_idx = model_input['glyph_idx']

        corner_embedding  = self.latent_code(instance_idx)
        glyph_embedding = self.glyph_code(glyph_idx)
        if self.extra_embedding:
            extra_embedding = self.extra_code(instance_idx)
            corner_embedding = torch.cat([corner_embedding, extra_embedding], dim=0)
            coords_corner = {'coords': torch.cat([model_input['coords'], model_input['excoords']], dim=0)}
        
        hypo_cf1 = self.hyper_corner_net1(corner_embedding)
        hypo_glyph1 = self.hyper_glyph_net1(glyph_embedding)
        hypo_cf2 = self.hyper_corner_net2(corner_embedding)
        hypo_glyph2 = self.hyper_glyph_net2(glyph_embedding)
        hypos1 = OrderedDict()
        hypos2 = OrderedDict()
        for key in hypo_cf1.keys():
            hypos1[key] = hypo_cf1[key] + hypo_glyph1[key]
        for key in hypo_cf2.keys():
            hypos2[key] = hypo_cf2[key] + hypo_glyph2[key]
        output_mids = self.corner_net1(coords_corner, params=hypos1)
        output_corner = self.corner_net2({'coords': output_mids['model_out']}, params=hypos2)
        
        model_out = {
            'corner_out': output_corner['model_out'], 
            'corner_vec': corner_embedding,
        }
        losses = self.corner_loss(model_out, model_input)
        return losses 

    def corner_loss(self, model_output, model_input):
        # --- --- ---
        gt_corner = model_input['probs'].unsqueeze(2)
        gt_sdf = model_input['sdflow'].unsqueeze(2)
        instance_idx = model_input['instance_idx']
        batchsize, n_sample, _ = gt_corner.shape

        if self.extra_embedding:            
            gdev = gt_corner.get_device()
            # gt_extra = model_input['probs'].unsqueeze(2) * torch.rand((batchsize, 1, 1), device=gdev).expand(batchsize, n_sample, 1)
            gt_extra = model_input['exprobs']
            gt_extra = self.extra_coe[instance_idx].unsqueeze(1).expand(*gt_extra.shape) * gt_extra
            gt_extra = gt_extra.unsqueeze(2)
            gt_corner = torch.cat([gt_corner, gt_extra], dim=0)
        

        pred_corner = model_output['corner_out'][:, :, :1]
        pred_sdf = model_output['corner_out'][:batchsize, :, 1:]
        corner_embeddings = model_output['corner_vec']

        # import pdb; pdb.set_trace()

        corner_constraint = torch.binary_cross_entropy_with_logits(pred_corner, gt_corner)
        corner_constraint = corner_constraint
        # amplify the maxima
        peak_constraint = corner_constraint * (gt_corner ** 2)
        # embeddings_constraint = torch.mean(corner_embeddings ** 2)
        flow_constraint = torch.abs(pred_sdf-gt_sdf)

        return {
                'corner': torch.abs(corner_constraint).mean() * 3e3,
                'peak': torch.abs(peak_constraint).mean() * 3e3,
                # 'embeddings_constraint': embeddings_constraint.mean() * 1e3,
                # 'sdflow': flow_constraint.mean() * 3e3
        }

