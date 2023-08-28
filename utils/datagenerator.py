
from torch.utils.data import DataLoader,Dataset

from PIL import Image
import numpy as np
import cv2
from utils.svgutils import *
from utils.svgutilsXtend import *
import pickle
import os
from utils.svgrefine import *


def render_png(wrpath, sequence, seqlen, refine = True, coe = 1.2, save_png = True):
    if refine:
        z, tlr, tud = reshape_svg(sequence, seqlen, coe = coe)
        y=mat2path(z, seqlen, 1024, [tlr*1024, tud*1024])
    else:
        y=mat2path(sequence, seqlen, 1024)
    if (save_png):
        cairosvg.svg2png(bytestring=y, write_to=wrpath ,background_color='#FFFFFFFF')
    return read_png(wrpath), z, tlr, tud

def sample_on_sdf(img, render_folder, wrpath, prefix, z, seqlen, tlr, tud, stri, strj, num_sample=5000):
    sdfvalue = compute_sdf(img)
    free_rand_idcs = np.random.uniform(0, 1, size=(num_sample, 2))
    free_rand_idcs_d = (free_rand_idcs*(1024-1)).astype(np.long)
    sdfvalue = sdfvalue[free_rand_idcs_d[:, 0], free_rand_idcs_d[:, 1]]
    
    np.save(os.path.join(render_folder, f'{prefix}{strj}/sdfs', f'{stri}.npy'), np.concatenate([sdfvalue.reshape(num_sample, 1), free_rand_idcs], axis=1))

    sampj, normals = sample_svg(z, seqlen, num_sample, [tlr, tud])
    np.save(os.path.join(render_folder, f'{prefix}{strj}/samps', f'{stri}sampas.npy'), sampj)
    im = cv2.imread(wrpath, cv2.IMREAD_GRAYSCALE)
    # import pdb; pdb.set_trace()
    outer = (sampj*1024 + normals*2).astype(np.long)
    try:
        rev_normals = (im[outer[:,0], outer[:,1]] == 0)
    except:
        return
    normals[rev_normals] = -normals[rev_normals]
    np.save(os.path.join(render_folder, f'{prefix}{strj}/normals', f'{stri}normals.npy'), normals)

resolutions = np.array([i/1024. for i in range(1024)], dtype=np.float32)
resolutions_x = np.expand_dims(resolutions, 0).repeat(1024, 0).flatten()
resolutions_y = np.expand_dims(resolutions, 1).repeat(1024, 1).flatten()
def sample_on_cornermap(img, render_folder, wrpath, prefix, z, seqlen, tlr, tud, stri, strj, perturb = False):
    corners = gen_corners(z, seqlen, [tlr, tud])
    if perturb:
        corners += np.random.rand(*corners.shape)*0.1
    lc = len(corners)
    if (lc == 0):
        free_rands = np.random.choice(1024**2, 5000)
        output_x = resolutions_x[free_rands]
        output_y = resolutions_y[free_rands]
        output_probs = np.zeros_like(output_x)
        save_cmap = np.stack([output_y, output_x, output_probs], axis = 0)
        np.save(os.path.join(render_folder, f'{prefix}{strj}/cmaps', f'{stri}cmap.npy'), save_cmap)
        np.save(os.path.join(render_folder, f'{prefix}{strj}/dcmaps', f'{stri}dcmap.npy'), save_cmap)
        return 
    mu_x = corners[:, 0]
    mu_y = corners[:, 1]
    sigma = 0.05
    corner_map = np.exp(-((resolutions_x.reshape(-1,1) - mu_x.reshape(1,-1)) ** 2 +
        (resolutions_y.reshape(-1,1) - mu_y.reshape(1,-1)) ** 2) / (2 * sigma ** 2)).max(1)
    # Data Part
    free_rands = np.random.choice(1024**2, 5000)
    output_x = resolutions_x[free_rands]
    output_y = resolutions_y[free_rands]
    output_probs = corner_map[free_rands]
    # reverse xy
    save_cmap = np.stack([output_y, output_x, output_probs], axis = 0)
    np.save(os.path.join(render_folder, f'{prefix}{strj}/cmaps', f'{stri}cmap.npy'), save_cmap)

    # Data Part
    field_rands_x = (np.random.rand(100*lc, 1)-0.5)/10 
    field_rands_y = (np.random.rand(100*lc, 1)-0.5)/10
    for i in range(lc):
        field_rands_x[i*100:(i+1)*100] += mu_x[i]
        field_rands_y[i*100:(i+1)*100] += mu_y[i]

    dense_corner_map = np.exp(-((field_rands_x - mu_x.reshape(1,-1)) ** 2 +
        (field_rands_y - mu_y.reshape(1,-1)) ** 2) / (2 * sigma ** 2)).max(1)
    # reverse xy
    save_dcmap = np.stack([field_rands_y.squeeze(1), field_rands_x.squeeze(1), dense_corner_map], axis = 0)
    np.save(os.path.join(render_folder, f'{prefix}{strj}/dcmaps', f'{stri}dcmap.npy'), save_dcmap)
    
    # Data Part
    whole_dense_corner_map = np.exp(-((resolutions_x.reshape(-1, 1) - mu_x.reshape(1,-1)) ** 2 +
        (resolutions_y.reshape(-1, 1) - mu_y.reshape(1,-1)) ** 2) / (2 * sigma ** 2)).max(1)
    whole_dense_pic = (whole_dense_corner_map.reshape(1024, 1024) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(render_folder, f'{prefix}{strj}/fpic', f'{stri}.png'), whole_dense_pic)

    # Data Part
    np.save(os.path.join(render_folder, f'{prefix}{strj}/corners', f'{stri}corner.npy'), corners)



def generate_csdflow(img, render_folder, wrpath, prefix, z, seqlen, tlr, tud, stri, strj):
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{strj}/cmaps', f'{stri}cmap.npy')):
        return
    sdfvalue = compute_sdf(img)
    cmap = np.load(os.path.join(render_folder, f'{prefix}{strj}/cmaps', f'{stri}cmap.npy'))
    dcmap = np.load(os.path.join(render_folder, f'{prefix}{strj}/dcmaps', f'{stri}dcmap.npy'))
    
    cmap_d = (cmap*(1024-1)).astype(np.long)
    dcmap_d = (dcmap*(1024-1)).astype(np.long)
    csdflow = sdfvalue[cmap_d[0, :], cmap_d[1, :]]
    dcsdflow = sdfvalue[dcmap_d[0, :], dcmap_d[1, :]]

    np.save(os.path.join(render_folder, f'{prefix}{strj}/csdflow', f'{stri}.npy'), csdflow)
    np.save(os.path.join(render_folder, f'{prefix}{strj}/dcsdflow', f'{stri}.npy'), dcsdflow)


def render_corner_field(render_folder, num_renders, j, prefix='Alphabet', data_PATH = '../test_all.pkl', trainset = None):
    if trainset is None:
        with open(data_PATH, 'rb') as f:
            trainset = pickle.load(f)
    if not os.path.exists(render_folder):
        os.makedirs(render_folder)
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{j}')):
        os.makedirs(os.path.join(render_folder, f'{prefix}{j}'))
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{j}/sdfs')):
        os.makedirs(os.path.join(render_folder, f'{prefix}{j}/sdfs'))
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{j}/samps')):
        os.makedirs(os.path.join(render_folder, f'{prefix}{j}/samps'))
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{j}/normals')):
        os.makedirs(os.path.join(render_folder, f'{prefix}{j}/normals'))
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{j}/cmaps')):
        os.makedirs(os.path.join(render_folder, f'{prefix}{j}/cmaps'))
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{j}/dcmaps')):
        os.makedirs(os.path.join(render_folder, f'{prefix}{j}/dcmaps'))
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{j}/corners')):
        os.makedirs(os.path.join(render_folder, f'{prefix}{j}/corners'))
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{j}/fpic')):
        os.makedirs(os.path.join(render_folder, f'{prefix}{j}/fpic'))
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{j}/csdflow')):
        os.makedirs(os.path.join(render_folder, f'{prefix}{j}/csdflow'))
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{j}/dcsdflow')):
        os.makedirs(os.path.join(render_folder, f'{prefix}{j}/dcsdflow'))

    for i in tqdm(range(len(trainset))):
        sequence = trainset[i]['sequence'][j]
        seqlen = trainset[i]['seq_len'][j][0]
        fname = f'{i}.png'
        wrpath = os.path.join(render_folder, f'{prefix}{j}', fname)
        img, z, tlr, tud = render_png(wrpath, sequence, seqlen, save_png = True)
        sample_on_cornermap(img, render_folder, wrpath, prefix, z, seqlen, tlr, tud, i, j, perturb=False)
        sample_on_sdf(img, render_folder, wrpath, prefix, z, seqlen, tlr, tud, i, j, num_sample=5000)
        generate_csdflow(img, render_folder, wrpath, prefix, z, seqlen, tlr, tud, i, j)
    return trainset

def render_svg(wrpath, sequence, seqlen, refine = True, coe = 1.2, save_png = True):
    if refine:
        z, tlr, tud = reshape_svg(sequence, seqlen, coe = coe)
        y=mat2path(z, seqlen, 1024, [tlr*1024, tud*1024])
    else:
        y=mat2path(sequence, seqlen, 1024)
    with open(wrpath, 'w') as f:
        f.write(y)
    

def render_svgs(render_folder, num_renders, j, prefix='Alphabet', data_PATH = '../test_all.pkl', trainset = None):
    if trainset is None:
        with open(data_PATH, 'rb') as f:
            trainset = pickle.load(f)
    if not os.path.exists(render_folder):
        os.makedirs(render_folder)
    if not os.path.exists(os.path.join(render_folder, f'{prefix}{j}')):
        os.makedirs(os.path.join(render_folder, f'{prefix}{j}'))
    for i in tqdm(range(len(trainset))):
        sequence = trainset[i]['sequence'][j]
        seqlen = trainset[i]['seq_len'][j][0]
        fname = f'{i}.svg'
        wrpath = os.path.join(render_folder, f'{prefix}{j}', fname)
        render_svg(wrpath, sequence, seqlen, save_png = True)
    return trainset