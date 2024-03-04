from torch.utils import data
from torch.utils.data import DataLoader,Dataset

from PIL import Image
import numpy as np
import cv2
import pickle
import os
from tqdm import tqdm
from utils.svgutils import compute_sdf
from math import sqrt

class BasicLoader(Dataset):
    def __init__(self, nfont_style, root_dir = None):
        self.root_dir = root_dir
        self.nfont_style = nfont_style
        self.font_idx = []

    def get_sdf(self, style):
        fnpy = os.path.join(self.root_dir, f'sdfs/{style}.npy')
        return np.load(fnpy)

    def get_sampling(self, style):
        fnpy = os.path.join(self.root_dir, f'samps/{style}sampas.npy')
        return np.load(fnpy)

    def get_normals(self, style):
        fnpy = os.path.join(self.root_dir, f'normals/{style}normals.npy')
        return np.load(fnpy)

    def load_corner_field(self, style):
        fnpy = os.path.join(self.root_dir, f'cmaps/{style}cmap.npy')
        return np.load(fnpy)

    def load_corner_brim(self, style):
        fnpy = os.path.join(self.root_dir, f'dcmaps/{style}dcmap.npy')
        return np.load(fnpy)

    def get_img(self, style):
        fimg = os.path.join(self.root_dir, f'{style}.png')
        img = cv2.imread(fimg, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        return img

    def get_img_uint8(self, style, resolution):
        fimg = os.path.join(self.root_dir, f'{style}.png')
        img = cv2.imread(fimg, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        return img
    
    def get_fpic(self, style, resolution):
        fimg = os.path.join(self.root_dir, f'fpic/{style}.png')
        img = cv2.imread(fimg, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        return img
    
    def get_corners_center(self, style):
        fnpy = np.load(os.path.join(self.root_dir, f'corners/{style}corner.npy'))
        np.random.shuffle(fnpy)
        cnpy = np.zeros((8, 2), dtype=np.float32)
        if (len(fnpy) > len(cnpy)):
            cnpy[:] = fnpy[:8]
        else:
            cnpy[:len(fnpy)] = fnpy[:]
        return cnpy

class CornerLoader(Dataset):
    def __init__(self, nfont_style, root_dir = None, extra_dir = None, mini_batch = 8):
        self.root_dir = root_dir
        self.nfont_style = nfont_style
        self.minibatch = mini_batch
        self.extra_dir = extra_dir

        self.font_idx = []
        
        self.nSample_cmap = 1000
        self.nSample_dcmap = 500

        for i in range(self.nfont_style):
            if os.path.exists(os.path.join(root_dir, f'cmaps/{i}cmap.npy')):
                self.font_idx.append(i)
        self.load_into_memory()
        self.length = len(self.font_idx) * self.minibatch

    def __len__(self):
        return self.length

    def load_corner_field(self, style):
        fnpy = os.path.join(self.root_dir, f'cmaps/{style}cmap.npy')
        return np.load(fnpy)

    def load_corner_brim(self, style):
        fnpy = os.path.join(self.root_dir, f'dcmaps/{style}dcmap.npy')
        return np.load(fnpy)

    def load_into_memory(self):
        self.cmaps = []
        self.dcmaps = []
        self.csdflows = []
        self.dcsdflows = []
        print('load the data into memory ... ')
        # print(self.font_idx)
        new_font_idx = []
        for i in tqdm(self.font_idx):
            csdflow = np.load(os.path.join(self.root_dir, f'csdflow/{i}.npy'))/1024.
            if (csdflow.max() > 100):
                continue
            self.cmaps.append(self.load_corner_field(i))
            self.dcmaps.append(self.load_corner_brim(i))
            self.csdflows.append(csdflow)
            self.dcsdflows.append(np.load(os.path.join(self.root_dir, f'dcsdflow/{i}.npy'))/1024.)
            new_font_idx.append(i)

        self.font_idx = new_font_idx

        if self.extra_dir is not None:
            self.excmaps = []
            self.exdcmaps = []
            for i in tqdm(self.font_idx):
                self.excmaps.append(np.load(os.path.join(self.extra_dir, f'cmaps/{i}cmap.npy')))
                self.exdcmaps.append(np.load(os.path.join(self.extra_dir, f'dcmaps/{i}dcmap.npy')))
        # print(np.array(checkz).max())
        # import pdb; pdb.set_trace()
            

    def __getitem__(self, index):
        absolute_index = index // self.minibatch
        style_index = self.font_idx[absolute_index]

        cmap = self.cmaps[absolute_index]
        dcmap = self.dcmaps[absolute_index]
        csdflow = self.csdflows[absolute_index]
        dcsdflow = self.dcsdflows[absolute_index]

        sample_cmap_idcs = np.random.choice(cmap.shape[1], self.nSample_cmap)
        sample_dcmap_idcs = np.random.choice(dcmap.shape[1], self.nSample_dcmap)

        sample_cmap = cmap[:, sample_cmap_idcs]
        sample_dcmap = dcmap[:, sample_dcmap_idcs]

        sample_all = np.concatenate([sample_cmap, sample_dcmap], axis = 1)

        coords = sample_all[:2, :].transpose(1, 0)
        probs = sample_all[2, :]
        sdflow = np.concatenate([csdflow[sample_cmap_idcs], dcsdflow[sample_dcmap_idcs]], axis=-1)

        Cdata = {
            'instance_idx': style_index,
            'coords': coords.astype(np.float32),
            'probs': probs.astype(np.float32),
            'sdflow': sdflow.astype(np.float32),
        }

        if self.extra_dir is not None:
            excmap = self.excmaps[absolute_index]
            exdcmap = self.exdcmaps[absolute_index]
            sample_excmap = excmap[:, sample_cmap_idcs]
            sample_exdcmap = exdcmap[:, sample_dcmap_idcs]

            sample_exall = np.concatenate([sample_excmap, sample_exdcmap], axis = 1)
            excoords = sample_exall[:2, :].transpose(1, 0)
            exprobs = sample_exall[2, :]

            Cdata['excoords'] = excoords.astype(np.float32)
            Cdata['exprobs'] =  exprobs.astype(np.float32)

        return Cdata

class SingleImageSDF(Dataset):
    def __init__(self,
                 root_dir,
                 glyph,
                 style):
        self.root_dir = root_dir
        # self.img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        # from utils.svgutils import compute_sdf
        # self.sdf = compute_sdf(self.img)
        fnpy = os.path.join(self.root_dir, f'sdfs/{style}.npy')
        self.sdfs = np.load(fnpy)
        self.sdfs = self.sdfs[(self.sdfs<0.5)[:,1]]
        self.sdfs[:, 0] /= 1024
        self.sdfs = self.sdfs.astype(np.float32)
        fnpy = os.path.join(self.root_dir, f'samps/{style}sampas.npy')
        self.samples = np.load(fnpy)
        mask_samples = (self.samples<0.5)[:,0]
        self.samples = self.samples[mask_samples].astype(np.float32)
        fnpy = os.path.join(self.root_dir, f'normals/{style}normals.npy')
        self.normals = np.load(fnpy)
        self.normals = self.normals[mask_samples].astype(np.float32)
        # import pdb; pdb.set_trace()
        self.nSample_edge = 1000
        self.nSample_free = 1000
        self.sample_replace = False
        self.style_index = style
        self.glyph_index = glyph

    def __len__(self):
        return 1
    def __getitem__(self, index):
        sdfdata = self.sdfs
        contdata = self.samples
        normaldata = self.normals

        rand_idcs = np.random.choice(contdata.shape[0], size=self.nSample_edge, replace=self.sample_replace)
        free_rand_idcs = np.random.choice(sdfdata.shape[0], size=self.nSample_free, replace=self.sample_replace)

        edgesamples = (contdata[rand_idcs]).astype(np.float32)
        edgesamples = np.clip(edgesamples, 0, 1.)

        normal = np.ones((self.nSample_free+self.nSample_edge, 2), dtype=np.float32) * -1
        normal[:self.nSample_edge] = normaldata[rand_idcs]
        sdfvalue = np.zeros((self.nSample_free+self.nSample_edge, ), dtype=np.float32)

        sdfvalue[self.nSample_edge:] = sdfdata[free_rand_idcs, 0]
        coordinate = np.concatenate([edgesamples, sdfdata[free_rand_idcs, 1:]], axis=0)

        b_normal = np.zeros((normal.shape[0], ), dtype=np.bool)
        b_normal[:self.nSample_edge] = True
        
        im_index = np.array([self.style_index], dtype=np.int64)
        # print(coordinate)
        return {'coords': coordinate, 
        'sdf': sdfvalue, 
        'normal': normal,
        'instance_idx': im_index,
        'b_normal': b_normal,
        'glyph_idx': np.array([self.glyph_index], dtype=np.int64)
        }

class SdfLoader(Dataset):
    def __init__(self, 
        nfont_style, 
        root_dir = None, 
        minibatch = 8,
        nSample_load = 5000):

        self.root_dir = root_dir
        self.nfont_style = nfont_style
        self.minibatch = minibatch
        
        self.font_idx = []
        
        self.nSample_edge = 1000
        self.nSample_free = 1000
        self.sample_replace = False
        self.nSample_load = nSample_load

        for i in range(self.nfont_style):
            if os.path.exists(os.path.join(root_dir, f'normals/{i}normals.npy')):
                self.font_idx.append(i)
        self.load_into_memory(self.nfont_style)
        self.length = len(self.font_idx) * self.minibatch

    def __len__(self):
        return self.length

    def get_sdf(self, style):
        fnpy = os.path.join(self.root_dir, f'sdfs/{style}.npy')
        return np.load(fnpy)

    def get_sampling(self, style):
        fnpy = os.path.join(self.root_dir, f'samps/{style}sampas.npy')
        return np.load(fnpy)

    def get_normals(self, style):
        fnpy = os.path.join(self.root_dir, f'normals/{style}normals.npy')
        return np.load(fnpy)


    def load_into_memory(self, len_data):
        self.sdfs = np.zeros((len_data, self.nSample_load, 3), dtype=np.float32)
        self.samples = np.zeros((len_data, self.nSample_load, 2), dtype=np.float32)
        self.normals = np.zeros((len_data, self.nSample_load, 2), dtype=np.float32)
        print('load the data into memory ... ')
        new_font_idx = []
        for i in tqdm(self.font_idx):
            self.sdfs[i] = self.get_sdf(i)
            self.sdfs[i][:, 0] /= (1024.)
            if self.sdfs[i][0, 0] < 100:
                new_font_idx.append(i)
            self.samples[i] = self.get_sampling(i)
            self.normals[i] = self.get_normals(i)
        self.font_idx = new_font_idx
            

    def __getitem__(self, index):
        absolute_index = index // self.minibatch
        style_index = self.font_idx[absolute_index]
        batch_index = index % self.minibatch

        sdfdata = self.sdfs[style_index]
        contdata = self.samples[style_index]
        normaldata = self.normals[style_index]

        rand_idcs = np.random.choice(contdata.shape[0], size=self.nSample_edge, replace=self.sample_replace)
        free_rand_idcs = np.random.choice(sdfdata.shape[0], size=self.nSample_free, replace=self.sample_replace)

        edgesamples = (contdata[rand_idcs]).astype(np.float32)
        edgesamples = np.clip(edgesamples, 0, 1.)

        normal = np.ones((self.nSample_free+self.nSample_edge, 2), dtype=np.float32) * -1
        normal[:self.nSample_edge] = normaldata[rand_idcs]
        sdfvalue = np.zeros((self.nSample_free+self.nSample_edge, ), dtype=np.float32)

        sdfvalue[self.nSample_edge:] = sdfdata[free_rand_idcs, 0]
        coordinate = np.concatenate([edgesamples, sdfdata[free_rand_idcs, 1:]], axis=0)

        b_normal = np.zeros((normal.shape[0], ), dtype=np.bool)
        b_normal[:self.nSample_edge] = True
        
        im_index = np.array([style_index], dtype=np.int64)
        # print(coordinate)
        return {'coords': coordinate, 
        'sdf': sdfvalue, 
        'normal': normal,
        'instance_idx': im_index,
        'b_normal': b_normal
        }

class CombinedLoader(BasicLoader):
    def __init__(self, nfont_style, root_dir = None):
        super().__init__(nfont_style, root_dir)
        self.minibatch = 8
        
        self.nSample_edge = 1000
        self.nSample_free = 1000
        self.nSample_cmap = 1000
        self.nSample_dcmap = 1000

        for i in range(self.nfont_style):
            if os.path.exists(os.path.join(root_dir, f'normals/{i}normals.npy')) \
            and os.path.exists(os.path.join(root_dir, f'cmaps/{i}cmap.npy')):
                self.font_idx.append(i)
        
        self.load_into_memory(self.nfont_style)
        self.length = len(self.font_idx) * self.minibatch

    def __len__(self):
        return self.length

    def load_into_memory(self, len_data):
        self.cmaps = []
        self.dcmaps = []
            
        self.sdfs = np.zeros((len_data, 5000, 3), dtype=np.float32)
        self.samples = np.zeros((len_data, 5000, 2), dtype=np.float32)
        self.normals = np.zeros((len_data, 5000, 2), dtype=np.float32)
        print('load the data into memory ... ')
        new_font_idx = []
        for i in tqdm(self.font_idx):
            self.sdfs[i] = self.get_sdf(i)
            self.sdfs[i][:, 0] /= 1024.
            if self.sdfs[i][0, 0] < 100:
                new_font_idx.append(i)
                self.cmaps.append(self.load_corner_field(i))
                self.dcmaps.append(self.load_corner_brim(i))
            self.samples[i] = self.get_sampling(i)
            self.normals[i] = self.get_normals(i)
        self.font_idx = new_font_idx

    def __getitem__(self, index):
        absolute_index = index // self.minibatch
        style_index = self.font_idx[absolute_index]
        batch_index = index % self.minibatch

        sdfdata = self.sdfs[style_index]
        contdata = self.samples[style_index]
        normaldata = self.normals[style_index]

        rand_idcs = np.random.choice(contdata.shape[0], size=self.nSample_edge)
        free_rand_idcs = np.random.choice(sdfdata.shape[0], size=self.nSample_free)

        edgesamples = (contdata[rand_idcs]).astype(np.float32)
        edgesamples = np.clip(edgesamples, 0, 1.)

        normal = np.ones((self.nSample_free+self.nSample_edge, 2), dtype=np.float32) * -1
        normal[:self.nSample_edge] = normaldata[rand_idcs]
        sdfvalue = np.zeros((self.nSample_free+self.nSample_edge, ), dtype=np.float32)

        sdfvalue[self.nSample_edge:] = sdfdata[free_rand_idcs, 0]
        coordinate = np.concatenate([edgesamples, sdfdata[free_rand_idcs, 1:]], axis=0)

        b_normal = np.zeros((normal.shape[0], ), dtype=np.bool)
        b_normal[:self.nSample_edge] = True
        

        im_index = np.array([style_index], dtype=np.int64)

        cmap = self.cmaps[absolute_index]
        dcmap = self.dcmaps[absolute_index]

        sample_cmap_idcs = np.random.choice(cmap.shape[1], self.nSample_cmap)
        sample_dcmap_idcs = np.random.choice(dcmap.shape[1], self.nSample_dcmap)

        sample_cmap = cmap[:, sample_cmap_idcs]
        sample_dcmap = dcmap[:, sample_dcmap_idcs]

        sample_all = np.concatenate([sample_cmap, sample_dcmap], axis = 1)

        coords = sample_all[:2, :].transpose(1, 0)
        probs = sample_all[2, :]

        return {'coords_sdf': coordinate, 
        'sdf': sdfvalue, 
        'normal': normal,
        'instance_idx': im_index,
        'b_normal': b_normal,
        'coords_corner': coords.astype(np.float32),
        'corner': probs.astype(np.float32),
        'corner_coords': self.get_corners_center(style_index),
        # 'imgs': self.get_img(style_index),
        # 'imgs': (compute_sdf(self.get_img_uint8(style_index, 256)) / 1024.).astype(np.float32),
        # 'fpic': self.get_fpic(style_index, 256)
        }

class ConvLoader(BasicLoader):
    def __init__(self, nfont_style, root_dir=None):
        super().__init__(nfont_style, root_dir)
        self.minibatch = 8
        
        self.nSample_edge = 1000
        self.nSample_free = 1000
        self.nSample_cmap = 1000
        self.nSample_dcmap = 1000

        for i in range(self.nfont_style):
            if os.path.exists(os.path.join(root_dir, f'normals/{i}normals.npy')) \
            and os.path.exists(os.path.join(root_dir, f'cmaps/{i}cmap.npy')):
                self.font_idx.append(i)
        self.length = len(self.font_idx) * self.minibatch
        self.load_into_memory(self.nfont_style)

    def __len__(self):
        return self.length

    def load_into_memory(self, len_data):
        self.sdfs = np.zeros((len_data, 5000, 3), dtype=np.float32)
        self.samples = np.zeros((len_data, 5000, 2), dtype=np.float32)
        self.normals = np.zeros((len_data, 5000, 2), dtype=np.float32)
        print('load the data into memory ... ')
        new_font_idx = []
        for i in tqdm(self.font_idx):
            self.sdfs[i] = self.get_sdf(i)
            self.sdfs[i][:, 0] /= 1024.
            if self.sdfs[i][0, 0] < 100:
                new_font_idx.append(i)
            self.samples[i] = self.get_sampling(i)
            self.normals[i] = self.get_normals(i)
        self.font_idx = new_font_idx

    def __getitem__(self, index):
        absolute_index = index // self.minibatch
        style_index = self.font_idx[absolute_index]
        batch_index = index % self.minibatch

        sdfdata = self.sdfs[style_index]
        contdata = self.samples[style_index]
        normaldata = self.normals[style_index]

        rand_idcs = np.random.choice(contdata.shape[0], size=self.nSample_edge)
        free_rand_idcs = np.random.choice(sdfdata.shape[0], size=self.nSample_free)

        edgesamples = (contdata[rand_idcs]).astype(np.float32)
        edgesamples = np.clip(edgesamples, 0, 1.)

        normal = np.ones((self.nSample_free+self.nSample_edge, 2), dtype=np.float32) * -1
        normal[:self.nSample_edge] = normaldata[rand_idcs]
        sdfvalue = np.zeros((self.nSample_free+self.nSample_edge, ), dtype=np.float32)

        sdfvalue[self.nSample_edge:] = sdfdata[free_rand_idcs, 0]
        coordinate = np.concatenate([edgesamples, sdfdata[free_rand_idcs, 1:]], axis=0)

        b_normal = np.zeros((normal.shape[0], ), dtype=np.bool)
        b_normal[:self.nSample_edge] = True

        im_index = np.array([style_index], dtype=np.int64)

        return {'coords': coordinate, 
        'sdf': sdfvalue, 
        'normal': normal,
        'instance_idx': im_index,
        'b_normal': b_normal,
        'imgs': self.get_img(style_index)
        }

class FpicLoader(Dataset):
    def __init__(self, nfont_style, root_dir = None):
        self.root_dir = root_dir
        self.nfont_style = nfont_style
        
        self.font_idx = []        
        self.resolution = 256

        for i in range(self.nfont_style):
            if os.path.exists(os.path.join(root_dir, f'fpic/{i}.png')) and \
                os.path.exists(os.path.join(root_dir, f'{i}.png')):
                self.font_idx.append(i)
        self.length = len(self.font_idx)
        self.load_into_memory(self.nfont_style)

    def __len__(self):
        return self.length

    def get_fpic(self, style):
        fpng = os.path.join(self.root_dir, f'fpic/{style}.png')
        return cv2.imread(fpng, cv2.IMREAD_GRAYSCALE)
    
    def get_font(self, style):
        fpng = os.path.join(self.root_dir, f'{style}.png')
        return cv2.imread(fpng, cv2.IMREAD_GRAYSCALE)

    def load_into_memory(self, len_data):
        self.fpics = np.zeros((len_data, self.resolution, self.resolution), dtype=np.float32)
        self.fonts = np.zeros((len_data, self.resolution, self.resolution), dtype=np.float32)
        for i in tqdm(self.font_idx):
            self.fpics[i] = cv2.resize(self.get_fpic(i), (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
            self.fonts[i] = cv2.resize(self.get_font(i), (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        self.fpics /= 255.

    def __getitem__(self, index):
        style_index = self.font_idx[index]

        return {
            'instance_idx': style_index,
            'input_font': self.fonts[index],
            'gt': self.fpics[index]
        }
    
class OccupancyLoader(Dataset):
    def __init__(self, nfont_style, root_dir = None):
        self.root_dir = root_dir
        self.nfont_style = nfont_style
        self.minibatch = 8
        

        self.font_idx = []
        
        self.nSample_edge = 1000
        self.nSample_free = 1000

        for i in range(self.nfont_style):
            if os.path.exists(os.path.join(root_dir, f'normals/{i}normals.npy')):
                self.font_idx.append(i)
        self.load_into_memory(self.nfont_style)
        self.length = len(self.font_idx) * self.minibatch

    def __len__(self):
        return self.length

    def get_sdf(self, style):
        fnpy = os.path.join(self.root_dir, f'sdfs/{style}.npy')
        return np.load(fnpy)

    def get_sampling(self, style):
        fnpy = os.path.join(self.root_dir, f'samps/{style}sampas.npy')
        return np.load(fnpy)

    def get_normals(self, style):
        fnpy = os.path.join(self.root_dir, f'normals/{style}normals.npy')
        return np.load(fnpy)


    def load_into_memory(self, len_data):
        self.sdfs = np.zeros((len_data, 5000, 3), dtype=np.float32)
        self.samples = np.zeros((len_data, 5000, 2), dtype=np.float32)
        print('load the data into memory ... ')
        new_font_idx = []
        for i in tqdm(self.font_idx):
            self.sdfs[i] = self.get_sdf(i)
            self.sdfs[i][:, 0] /= 1024.
            if self.sdfs[i][0, 0] < 100:
                new_font_idx.append(i)
            self.samples[i] = self.get_sampling(i)
        self.font_idx = new_font_idx
            

    def __getitem__(self, index):
        absolute_index = index // self.minibatch
        style_index = self.font_idx[absolute_index]
        batch_index = index % self.minibatch

        sdfdata = self.sdfs[style_index]
        contdata = self.samples[style_index]

        rand_idcs = np.random.choice(contdata.shape[0], size=self.nSample_edge)
        free_rand_idcs = np.random.choice(sdfdata.shape[0], size=self.nSample_free)

        edgesamples = (contdata[rand_idcs]).astype(np.float32)
        edgesamples = np.clip(edgesamples, 0, 1.)

        occvalue = np.random.randn(self.nSample_free+self.nSample_edge).astype(np.float32)
        occvalue[self.nSample_edge:] = sdfdata[free_rand_idcs, 0]
        value1 = occvalue > 0
        occvalue[value1] = 1
        occvalue[~value1] = 0

        coordinate = np.concatenate([edgesamples, sdfdata[free_rand_idcs, 1:]], axis=0)

        
        im_index = np.array([style_index], dtype=np.int64)
        return {'coords': coordinate, 
        'occ': occvalue, 
        'instance_idx': im_index
        }

class GaussianLoader(Dataset):
    def __init__(self):
        with open('Datasets/gaussian.pkl', 'rb') as f:
            gdict = pickle.load(f)
        
        self.imgs = gdict['mat']
        self.probs = gdict['prob']
        self.length = len(self.imgs)
        print(self.length)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            'img': self.imgs[index],
            'prob': self.probs[index]
        }

class MultiSDFLoader(Dataset):
    def __init__(self, nfont_style, root_dirs = [], minibatch = 8):
        self.root_dirs = root_dirs
        self.SDFLoaders = [SdfLoader(nfont_style, rd, minibatch=minibatch) for rd in root_dirs]
        self.nfont_style = nfont_style

        self.length = 0
        self.lengths = [len(loader) for loader in self.SDFLoaders]
        for l in self.lengths:
            self.length += l
        
        self.glyph_idcs = np.zeros((self.length, ), dtype=np.int64)
        self.style_idcs = np.zeros((self.length, ), dtype=np.int64)

        count_gid = 0
        count_sid = 0
        for i in range(self.length):
            self.glyph_idcs[i] = count_gid
            self.style_idcs[i] = count_sid
            count_sid += 1
            if count_sid >= self.lengths[count_gid]:
                count_gid += 1
                count_sid = 0
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        glyph_idx = self.glyph_idcs[index]
        style_idx = self.style_idcs[index]
        content = self.SDFLoaders[glyph_idx][style_idx]
        content['glyph_idx'] = glyph_idx

        return content

class MultiCFLoader(Dataset):
    def __init__(self, nfont_style, root_dirs = [], minibatch = 8):
        self.root_dirs = root_dirs
        self.SDFLoaders = [CornerLoader(nfont_style, rd, extra_dir=None, mini_batch=minibatch) for rd in root_dirs]
        self.nfont_style = nfont_style

        self.length = 0
        self.lengths = [len(loader) for loader in self.SDFLoaders]
        for l in self.lengths:
            self.length += l
        
        self.glyph_idcs = np.zeros((self.length, ), dtype=np.int64)
        self.style_idcs = np.zeros((self.length, ), dtype=np.int64)

        count_gid = 0
        count_sid = 0
        for i in range(self.length):
            self.glyph_idcs[i] = count_gid
            self.style_idcs[i] = count_sid
            count_sid += 1
            if count_sid >= self.lengths[count_gid]:
                count_gid += 1
                count_sid = 0
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        glyph_idx = self.glyph_idcs[index]
        style_idx = self.style_idcs[index]
        content = self.SDFLoaders[glyph_idx][style_idx]
        content['glyph_idx'] = glyph_idx

        return content