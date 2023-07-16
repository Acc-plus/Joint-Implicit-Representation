import numpy as np
import os
import cv2
from tqdm import tqdm

def generate_from_corners(corners, render_folder, i_corner):
    lc = len(corners)
    if (lc == 0):
        return
    mu_x = corners[:, 0]
    mu_y = corners[:, 1]
    sigma = 0.05
    resolutions = np.array([i/1024. for i in range(1024)], dtype=np.float32)
    resolutions_x = np.expand_dims(resolutions, 0).repeat(1024, 0).flatten()
    resolutions_y = np.expand_dims(resolutions, 1).repeat(1024, 1).flatten()
    corner_map = np.exp(-((resolutions_x.reshape(-1,1) - mu_x.reshape(1,-1)) ** 2 +
        (resolutions_y.reshape(-1,1) - mu_y.reshape(1,-1)) ** 2) / (2 * sigma ** 2)).max(1)
    # Data Part
    free_rands = np.random.choice(1024**2, 5000)
    output_x = resolutions_x[free_rands]
    output_y = resolutions_y[free_rands]
    output_probs = corner_map[free_rands]

    # reverse xy
    save_cmap = np.stack([output_y, output_x, output_probs], axis = 0)
    np.save(os.path.join(render_folder, f'cmaps', f'{i_corner}cmap.npy'), save_cmap)

    # Data Part
    field_rands_x = (np.random.rand(100*lc, 1) ) 
    field_rands_y = (np.random.rand(100*lc, 1) ) 
    dense_corner_map = np.exp(-((field_rands_x - mu_x.reshape(1,-1)) ** 2 +
        (field_rands_y - mu_y.reshape(1,-1)) ** 2) / (2 * sigma ** 2)).max(1)
    # reverse xy
    save_dcmap = np.stack([field_rands_y.squeeze(1), field_rands_x.squeeze(1), dense_corner_map], axis = 0)
    np.save(os.path.join(render_folder, f'dcmaps', f'{i_corner}dcmap.npy'), save_dcmap)
    
    # Data Part
    whole_dense_corner_map = np.exp(-((resolutions_x.reshape(-1, 1) - mu_x.reshape(1,-1)) ** 2 +
        (resolutions_y.reshape(-1, 1) - mu_y.reshape(1,-1)) ** 2) / (2 * sigma ** 2)).max(1)
    whole_dense_pic = (whole_dense_corner_map.reshape(1024, 1024) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(render_folder, f'fpic', f'{i_corner}.png'), whole_dense_pic)

    # Data Part
    np.save(os.path.join(render_folder, f'corners', f'{i_corner}corner.npy'), corners)

def sample_corner_from_area(render_folder, num_renders):
    if not os.path.exists(render_folder):
        os.makedirs(render_folder)
    if not os.path.exists(render_folder):
        os.makedirs(render_folder)
    if not os.path.exists(os.path.join(render_folder, f'cmaps')):
        os.makedirs(os.path.join(render_folder, f'cmaps'))
    if not os.path.exists(os.path.join(render_folder, f'dcmaps')):
        os.makedirs(os.path.join(render_folder, f'dcmaps'))
    if not os.path.exists(os.path.join(render_folder, f'corners')):
        os.makedirs(os.path.join(render_folder, f'corners'))
    if not os.path.exists(os.path.join(render_folder, f'fpic')):
        os.makedirs(os.path.join(render_folder, f'fpic'))

    area1 = lambda c: 0.65 < c[0] and c[0] < 0.8 and 0.2 < c[1] and c[1] < 0.8
    area2 = lambda c: 0.2 < c[0] and c[0] < 0.35 and 0.2 < c[1] and c[1] < 0.8
    areas = [area1, area2]
    contains = [4, 4]
    num_areas = areas.__len__()

    for i in tqdm(range(num_renders)):
        
        contains_Cp = np.array(contains, dtype=np.long)
        for j in range(num_areas):
            contains_Cp[j] -= np.random.randint(0, 2) * 2
        corners = []
        while contains_Cp.sum() > 0:
            x, y = np.random.rand(2)
            for j in range(num_areas):
                if areas[j]([x, y]) and contains_Cp[j] > 0:
                    contains_Cp[j] -= 1
                    corners.append([x, y])
                    break
        generate_from_corners(np.array(corners), render_folder, i)

if __name__ == '__main__':
    sample_corner_from_area('Datasets/cAuxTest4', 4000)
