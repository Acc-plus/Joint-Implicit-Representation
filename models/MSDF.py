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


class MultiSDF(nn.Module):
    def __init__(self, 
    num_instances,
    num_glyph,
    latent_dim=128, 
    hyper_hidden_layers=2,
    hyper_hidden_features=256,
    hidden_layer=4,
    hidden_features=128, 
    glyph_dim=32):
        super().__init__()

        self.latent_dim = latent_dim
        self.glyph_dim = glyph_dim
        self.num_instances = num_instances
        self.num_glyph = num_glyph
        self.glyph_code = nn.Embedding(num_glyph, self.glyph_dim)
        nn.init.normal_(self.glyph_code.weight, mean=0, std=0.01)
        self.latent_code = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_code.weight, mean=0, std=0.01)

        self.sdf_net = modules.SingleBVPNet(type='relu',mode='mlp', 
                                          hidden_features=hidden_features, 
                                          num_hidden_layers=hidden_layer, 
                                          in_features=2,
                                          out_features=1)

        self.hyper_sdf_net = HyperNetwork(hyper_in_features=self.latent_dim, 
                                          hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, 
                                          hypo_module=self.sdf_net)
        
        self.hyper_glyph_net = HyperNetwork(hyper_in_features=self.glyph_dim, 
                                          hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, 
                                          hypo_module=self.sdf_net)
        

    def forward(self, model_input):

        instance_idx = model_input['instance_idx']
        glyph_idx = model_input['glyph_idx']

        coords_sdf = {'coords': model_input['coords']}

        sdf_embedding = self.latent_code(instance_idx)
        glyph_embedding = self.glyph_code(glyph_idx)
        hypo_sdf = self.hyper_sdf_net(sdf_embedding)
        hypo_glyph = self.hyper_glyph_net(glyph_embedding)
        
        hypos = OrderedDict()
        for key in hypo_sdf.keys():
            hypos[key] = hypo_sdf[key] + hypo_glyph[key]
        output_sdf = self.sdf_net(coords_sdf, params=hypos)

        grad_sdf = torch.autograd.grad(output_sdf['model_out'], [model_input['coords']], grad_outputs=torch.ones_like(output_sdf['model_out']), create_graph=True)[0]

        model_out = {
            'sdf_out': output_sdf['model_out'],
            'sdf_vec': sdf_embedding,
            'grad_sdf': grad_sdf,
            'normal': model_input['normal'],
            'b_normal': model_input['b_normal']
        }
        losses = csdf_loss(model_out, model_input)
        return losses 
    
    def infernce_embedding(self, model_input):
    
        glyph_idx = model_input['glyph_idx']

        coords_sdf = {'coords': model_input['coords']}

        sdf_embedding = model_input['font']
        glyph_embedding = self.glyph_code(glyph_idx)
        hypo_sdf = self.hyper_sdf_net(sdf_embedding)
        hypo_glyph = self.hyper_glyph_net(glyph_embedding)

        hypos = OrderedDict()
        for key in hypo_sdf.keys():
            hypos[key] = hypo_sdf[key] + hypo_glyph[key]
        output_sdf = self.sdf_net(coords_sdf, params=hypos)
        return output_sdf['model_out']

    def forward_embedding(self, model_input):
    
        instance_idx = model_input['instance_idx']
        glyph_idx = model_input['glyph_idx']

        coords_sdf = {'coords': model_input['coords']}

        sdf_embedding = model_input['font']
        glyph_embedding = self.glyph_code(glyph_idx)
        hypo_sdf = self.hyper_sdf_net(sdf_embedding)
        hypo_glyph = self.hyper_glyph_net(glyph_embedding)
        
        hypos = OrderedDict()
        for key in hypo_sdf.keys():
            hypos[key] = hypo_sdf[key] + hypo_glyph[key]
        output_sdf = self.sdf_net(coords_sdf, params=hypos)

        grad_sdf = torch.autograd.grad(output_sdf['model_out'], [model_input['coords']], grad_outputs=torch.ones_like(output_sdf['model_out']), create_graph=True)[0]

        # import pdb; pdb.set_trace()
        model_out = {
            'sdf_out': output_sdf['model_out'],
            'sdf_vec': sdf_embedding,
            'grad_sdf': grad_sdf,
            'normal': model_input['normal'],
            'b_normal': model_input['b_normal']
        }
        losses = csdf_loss(model_out, model_input)
        return losses 

    def set_glyph_idx(self, idx):
        self.glyph_idx = torch.LongTensor([idx]).to(device)

    def inference(self, coords, instance_idx, emb = None):
        coords = coords.unsqueeze(0)
        instance_idx = torch.LongTensor([instance_idx]).to(device)
        if emb is not None:
            embedding = emb
        else:
            embedding = self.latent_code(instance_idx)
        emebdding_glyph = self.glyph_code(self.glyph_idx)
        hypo_sdf = self.hyper_sdf_net(embedding)
        hypo_glyph = self.hyper_glyph_net(emebdding_glyph.unsqueeze(0))
        hypos = OrderedDict()
        for key in hypo_sdf.keys():
            hypos[key] = hypo_sdf[key] + hypo_glyph[key]
        output_sdf = self.sdf_net({'coords': coords}, params=hypos)
        return output_sdf['model_out']

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
        # embedding = torch.einsum('i,oj->ij', [c, embedding2]) + torch.einsum('i,oj->ij', [cr, embedding1])
        for i in range(num_frames):
            embedding[i] = c[i] * embedding2 + cr[i] * embedding1

        hypo_sdf = self.hyper_sdf_net(embedding)
        hypo_glyph = self.hyper_glyph_net(emebdding_glyph.expand(num_frames, self.glyph_dim))
        hypos = OrderedDict()
        for key in hypo_sdf.keys():
            hypos[key] = hypo_sdf[key] + hypo_glyph[key]
        output_sdf = self.sdf_net({'coords': coords}, params=hypos)
        return output_sdf['model_out']
    
    def inference_mid(self, coords, instance_idx1, instance_idx2, coe):
        coords = coords.unsqueeze(0)
        instance_idx1 = torch.LongTensor([instance_idx1]).to(device)
        instance_idx2 = torch.LongTensor([instance_idx2]).to(device)

        embedding1 = self.latent_code(instance_idx1)
        embedding2 = self.latent_code(instance_idx2)
        emebdding_glyph = self.glyph_code(self.glyph_idx)
        embedding = embedding1 * (1 - coe) + embedding2 * coe
        
        hypo_sdf = self.hyper_sdf_net(embedding)
        hypo_glyph = self.hyper_glyph_net(emebdding_glyph.unsqueeze(0))
        hypos = OrderedDict()
        for key in hypo_sdf.keys():
            hypos[key] = hypo_sdf[key] + hypo_glyph[key]
        output_sdf = self.sdf_net({'coords': coords}, params=hypos)
        return output_sdf['model_out']


def csdf_loss(model_output, model_input):
    # --- --- ---
    gt_sdf = model_input['sdf'].unsqueeze(2)
    

    pred_sdf = model_output['sdf_out']
    sdf_embeddings = model_output['sdf_vec']
    grad_sdf = model_output['grad_sdf']
    gt_normal = model_output['normal']
    b_normal = model_output['b_normal']

    sdf_constraint = pred_sdf-gt_sdf

    # embeddings_constraint = torch.mean(sdf_embeddings ** 2)
    nl_values = 1 - F.cosine_similarity(grad_sdf, gt_normal, dim=-1)
    normal_loss = torch.where(b_normal, nl_values, torch.zeros_like(nl_values))
    # inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    el_values = torch.abs(torch.norm(grad_sdf, p=None, dim=-1) - 1.)
    Eikonal_loss = torch.where(b_normal, el_values, torch.zeros_like(el_values))

    return {
            'sdf': torch.abs(sdf_constraint).mean() * 3e3, 
            # 'embeddings_constraint': embeddings_constraint.mean() * 1e4,
            'Eikonal': Eikonal_loss.mean() * 5,
            'normal': normal_loss.mean() * 1e2,
            # 'inter_constraint': inter_constraint.mean() * 5e2
    }


class MultiSDFSimple(nn.Module):
    def __init__(self, 
    num_instances,
    num_glyph,
    latent_dim=128, 
    hyper_hidden_layers=2,
    hyper_hidden_features=256,
    hidden_num=128, 
    num_hidden_layer=4,
    glyph_dim=32):
        super().__init__()

        self.latent_dim = latent_dim
        self.glyph_dim = glyph_dim
        self.num_instances = num_instances
        self.num_glyph = num_glyph
        self.glyph_code = nn.Embedding(num_glyph, self.glyph_dim)
        nn.init.normal_(self.glyph_code.weight, mean=0, std=0.01)
        self.latent_code = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_code.weight, mean=0, std=0.01)

        self.sdf_net = modules.SingleBVPNet(type='relu',mode='mlp', 
                                          hidden_features=hidden_num, 
                                          num_hidden_layers=num_hidden_layer, 
                                          in_features=2+latent_dim+glyph_dim,
                                          out_features=1)

    def forward(self, model_input):

        instance_idx = model_input['instance_idx']
        glyph_idx = model_input['glyph_idx'].unsqueeze(1)
        model_input['coords'].requires_grad_()
        
        sdf_embedding = self.latent_code(instance_idx).expand(64,2000,self.latent_dim)
        glyph_embedding = self.glyph_code(glyph_idx).expand(64,2000,self.glyph_dim)
        # import pdb; pdb.set_trace() 
        coords_sdf = {'coords': torch.cat([sdf_embedding, glyph_embedding, model_input['coords']], -1)}
        
        
        output_sdf = self.sdf_net(coords_sdf)

        grad_sdf = torch.autograd.grad(output_sdf['model_out'], [model_input['coords']], grad_outputs=torch.ones_like(output_sdf['model_out']), create_graph=True)[0]

        model_out = {
            'sdf_out': output_sdf['model_out'],
            'sdf_vec': sdf_embedding,
            'grad_sdf': grad_sdf,
            'normal': model_input['normal'],
            'b_normal': model_input['b_normal']
        }
        losses = csdf_loss(model_out, model_input)
        return losses 
    
    def set_glyph_idx(self, idx):
        self.glyph_idx = torch.LongTensor([idx]).to(device)

    def inference(self, coords, instance_idx, emb = None):
        coords = coords.unsqueeze(0)
        instance_idx = torch.LongTensor([instance_idx]).to(device)
        if emb is not None:
            embedding = emb
        else:
            embedding = self.latent_code(instance_idx)
        emebdding_glyph = self.glyph_code(self.glyph_idx)
        coords = coords.view(-1, 2)
        embedding = embedding.expand(coords.shape[0], embedding.shape[1])
        emebdding_glyph = emebdding_glyph.expand(coords.shape[0], emebdding_glyph.shape[1])
        # import pdb; pdb.set_trace()
        output_sdf = self.sdf_net({'coords': torch.cat([embedding, emebdding_glyph, coords], -1)})
        return output_sdf['model_out']