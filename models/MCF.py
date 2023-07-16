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

class MultiCF(nn.Module):
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
            
        self.corner_net=modules.SingleBVPNet(type=activation,mode='mlp', 
                                             hidden_features=hidden_num, 
                                             num_hidden_layers=num_hidden_layer, 
                                             in_features=2,
                                             out_features=1+self.sdflow_feature)

        self.hyper_corner_net = HyperNetwork(hyper_in_features=self.latent_dim, 
                                             hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, 
                                             hypo_module=self.corner_net)
        
        self.hyper_glyph_net = HyperNetwork(hyper_in_features=self.glyph_dim, 
                                             hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, 
                                             hypo_module=self.corner_net)
    def set_glyph_idx(self, idx):
        self.glyph_idx = torch.LongTensor([idx]).to(device)

    def inference(self, coords, instance_idx, emb_mode = None):
        coords = coords.unsqueeze(0)
        instance_idx = torch.LongTensor([instance_idx]).to(device)
        embedding = self.latent_code(instance_idx)
        emebdding_glyph = self.glyph_code(self.glyph_idx)
        hypo_cf = self.hyper_corner_net(embedding)
        hypo_glyph = self.hyper_glyph_net(emebdding_glyph.unsqueeze(0))
        hypos = OrderedDict()
        for key in hypo_cf.keys():
            hypos[key] = hypo_cf[key] + hypo_glyph[key]
        output_sdf = self.corner_net({'coords': coords}, params=hypos)
        return torch.sigmoid(output_sdf['model_out'][:, :, 0])

    def inference_multi(self, coords, instance_idx1, instance_idx2, num_frames, emb_mode = None):
        coords = coords.unsqueeze(0).expand(num_frames, -1, 2)
        instance_idx1 = torch.LongTensor([instance_idx1]).to(device)
        instance_idx2 = torch.LongTensor([instance_idx2]).to(device)
        # import pdb; pdb.set_trace()
        embedding1 = self.latent_code(instance_idx1)
        embedding2 = self.latent_code(instance_idx2)
        emebdding_glyph = self.glyph_code(self.glyph_idx)
        
        # import pdb; pdb.set_trace()
        c = torch.arange(0,1,1/num_frames).to(device)
        c[num_frames - 1] = 1.
        cr = 1. - c
        embedding = torch.zeros(num_frames, self.latent_dim).to(device)
        # embedding = torch.einsum('i,oj->ij', [c, embedding2]) + torch.einsum('i,oj->ij', [cr, embedding1])
        for i in range(num_frames):
            embedding[i] = c[i] * embedding2 + cr[i] * embedding1

        hypo_sdf = self.hyper_corner_net(embedding)
        hypo_glyph = self.hyper_glyph_net(emebdding_glyph.expand(num_frames, self.glyph_dim))
        hypos = OrderedDict()
        for key in hypo_sdf.keys():
            hypos[key] = hypo_sdf[key] + hypo_glyph[key]
        output_corner = self.corner_net({'coords': coords}, params=hypos)
        return torch.sigmoid(output_corner['model_out'][:, :, 0])

    def inference_mid(self, coords, instance_idx1, instance_idx2, coe, **kwargs):
        coords = coords.unsqueeze(0)
        instance_idx1 = torch.LongTensor([instance_idx1]).to(device)
        instance_idx2 = torch.LongTensor([instance_idx2]).to(device)

        embedding1 = self.latent_code(instance_idx1)
        embedding2 = self.latent_code(instance_idx2)
        emebdding_glyph = self.glyph_code(self.glyph_idx)
        embedding = embedding1 * (1 - coe) + embedding2 * coe
        
        hypo_sdf = self.hyper_corner_net(embedding)
        hypo_glyph = self.hyper_glyph_net(emebdding_glyph.unsqueeze(0))
        hypos = OrderedDict()
        for key in hypo_sdf.keys():
            hypos[key] = hypo_sdf[key] + hypo_glyph[key]
        output_sdf = self.corner_net({'coords': coords}, params=hypos)
        return torch.sigmoid(output_sdf['model_out'])
    
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
            
        hypo_corner = self.hyper_corner_net(corner_embedding)
        hypo_glyph = self.hyper_glyph_net(glyph_embedding)
        hypos = OrderedDict()
        for key in hypo_corner.keys():
            hypos[key] = hypo_corner[key] + hypo_glyph[key]
        output_corner = self.corner_net(coords_corner, params=hypos)

        # import pdb; pdb.set_trace()

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
                'sdflow': flow_constraint.mean() * 3e3
        }

class MultiCFSimple(nn.Module):
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
            
        self.corner_net=modules.SingleBVPNet(type=activation,mode='mlp', 
                                             hidden_features=hidden_num, 
                                             num_hidden_layers=num_hidden_layer, 
                                             in_features=2+latent_dim+glyph_dim,
                                             out_features=1+self.sdflow_feature)
    def forward(self, model_input):
    
        instance_idx = model_input['instance_idx'].unsqueeze(1)
        glyph_idx = model_input['glyph_idx'].unsqueeze(1)
        # import pdb; pdb.set_trace()
        corner_embedding  = self.latent_code(instance_idx).expand(64,1500,self.latent_dim)
        glyph_embedding = self.glyph_code(glyph_idx).expand(64,1500,self.glyph_dim)
        coords_corner = {'coords': torch.cat([corner_embedding, glyph_embedding, model_input['coords']], -1)}

        output_corner = self.corner_net(coords_corner)

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
                'sdflow': flow_constraint.mean() * 3e3
        }