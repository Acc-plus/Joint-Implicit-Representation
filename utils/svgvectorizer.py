import enum
from os import dup
import torch
import numpy as np
import math
import cv2
import vct

def vectorize_edge(ov, cst, cend, resolution = 1024., has_corner = True):
    return vct.vectorize_edge(ov, cst, cend, resolution, has_corner)

def Translate_Primitives(primitives, parameters, resolution = 1024):
    paths = []
    for prim, param in zip(primitives, parameters):
        param = param * resolution
        if prim == 0:
            paths.append(f"L {param[1]} {param[0]}")
        elif prim == 1:
            paths.append(f"Q {param[1]} {param[0]} {param[3]} {param[2]}")
    
    return ' '.join(paths) + '\n'


def Combine_Rendering(ordered_vertices, corners, nearest, normals, curv, connection, resolution = 1024):
    # print('vectorize start')
    num_min_contour = 25
    # insert cutting points to ordered vertices
    # ordered vertices represents recurrent vertices, which is a closed graph
    b_vert = np.ones((ordered_vertices.shape[0], ), np.bool)
    connection_i = 0
    temp_verts = []
    Primitives = []
    nearest_back = 0
    path_str = ''
    # assert connection.__len__() - 1 == 1
    if corners.shape[0] != 0:
        nearest, ind_c = np.unique(nearest, return_index=True)
        corners = corners[ind_c]

        old_nearest = np.array(nearest)
        # reselect_nearest(nearest, curv, connection, 512)
        dual_contouring(nearest, normals, ordered_vertices, connection, corners, 512, 45)
        # nearest = old_nearest

        nearest, ind_c = np.unique(nearest, return_index=True)
        corners = corners[ind_c]
    
    # import pdb; pdb.set_trace()
    while connection_i < connection.__len__() - 1:
        # Eliminate the contour contains too few points
        temp_verts.clear()
        # import pdb; pdb.set_trace()
        if connection[connection_i + 1] - connection[connection_i] < num_min_contour:
            # if nearest_back >= len(nearest):
            #     break
            
            if nearest_back < len(nearest) and nearest[nearest_back] < connection[connection_i + 1]:
                nearest_back += 1
            connection_i += 1
            continue
        if  nearest_back >= len(nearest) or nearest[nearest_back] >= connection[connection_i + 1]:
            
            vert_i = connection[connection_i]
            for ci in range(connection[connection_i], connection[connection_i + 1]):
                temp_verts.append(ordered_vertices[ci])
            temp_verts.append(temp_verts[0])
            # temp_verts.append(temp_verts[1])
            # import pdb; pdb.set_trace()
            primitives, parameters = vct.vectorize_wc(np.array(temp_verts), 
                resolution,
                1., 2.)
            # print('--smooth--only--')
            # import pdb; pdb.set_trace()
            parameters = parameters.reshape(-1, 4)
            # if primitives[0] == 1:
            #     path_str += f'M {parameters[0][1] * resolution} {parameters[0][0] * resolution} '
            # else:
            path_str += f'M {parameters[-1][3] * resolution} {parameters[-1][2] * resolution} '
            # path_str += f'M {temp_verts[0][1] * resolution} {temp_verts[0][0] * resolution} '
            # primitives = primitives[:-1]
            # parameters = parameters[:-1]
            path_str += Translate_Primitives(primitives, parameters, resolution)
            temp_verts.clear()
            connection_i += 1
            continue
        vert_i = connection[connection_i]
        nearest_i = nearest_back
        nearest_front = nearest_back
        path_str += f'M {corners[nearest_i][1] * resolution} {corners[nearest_i][0] * resolution} '
        while vert_i <= nearest[nearest_i]:
            vert_i += 1
        # print(vert_i)
        prev_nearest = nearest_i
        nearest_i += 1
        if nearest_i >= len(nearest) or nearest[nearest_i] >= connection[connection_i + 1]:
            nearest_i = nearest_front
            nearest_back = nearest_i
        vert_i -= 1
        if vert_i < 0:
            vert_i = connection[connection_i + 1] - 1
        while b_vert[vert_i] == True:
            # import pdb; pdb.set_trace()
            b_vert[vert_i] = False
            vert_i += 1
            if vert_i >= connection[connection_i + 1]:
                vert_i = connection[connection_i]
            temp_verts.append(ordered_vertices[vert_i])
            if vert_i == nearest[nearest_i]:
                cst = prev_nearest
                cend = nearest_i
                prev_nearest = nearest_i
                nearest_i += 1
                if nearest_i >= len(nearest) or nearest[nearest_i] >= connection[connection_i + 1]:
                    # import pdb; pdb.set_trace()
                    nearest_back = nearest_i
                    nearest_i = nearest_front
                    
                # import pdb; pdb.set_trace()
                # print('tvc')
                primitives, parameters = vct.vectorize_edge(np.array(temp_verts), 
                    corners[cst],
                    corners[cend], 
                    resolution,
                    2., 2)
                # import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
                parameters = parameters.reshape(-1, 4)
                path_str += Translate_Primitives(primitives, parameters, resolution)
                temp_verts.clear()
            # temp_verts.append(ordered_vertices[vert_i])
        connection_i += 1

    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{resolution}px" height="{resolution}px" style="-ms-transform: rotate(360deg); -webkit-transform: rotate(360deg); transform: rotate(360deg);" preserveAspectRatio="xMidYMid meet" viewBox="0 0 {resolution} {resolution}">'
    f'<path d="{path_str}" stroke-width="1.0" fill="rgb(0, 0, 0)" opacity="1.0"></path></svg>')

def extract_contour(img):
    img = torch.from_numpy(img)
    kernel = torch.ones((3,3))
    mask1 = img == 0
    mask2 = torch.conv2d(img.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=1) > 0
    mask = mask1 & mask2
    mask = mask.squeeze(0).squeeze(0).numpy()
    return mask

# device = torch.device('cuda:4')
def Edge_Partition(fpng, fcorners, resolution):
    output = np.zeros((resolution, resolution), dtype = np.uint8)
    png = cv2.imread(fpng, cv2.IMREAD_GRAYSCALE) # GRAYSCALE uint8
    sdf = np.array(png)
    png[0, :] = 255
    png[resolution-1, :] = 255
    png[:, 0] = 255
    png[:, resolution-1] = 255
    png[png > 128] = 255
    png[png <= 128] = 0
    png = png.astype(np.float32)
    mask = extract_contour(png)
    vct.check_pngs(mask, png)
    mask = extract_contour(png)
    resolutions = np.array([i/float(resolution) for i in range(resolution)], dtype=np.float32)
    resolutions_x = np.expand_dims(resolutions, 1).repeat(resolution, 1)
    resolutions_y = np.expand_dims(resolutions, 0).repeat(resolution, 0)
    idcs = np.array(range(resolution))
    idcs_x = np.expand_dims(idcs, 1).repeat(resolution, 1)
    idcs_y = np.expand_dims(idcs, 0).repeat(resolution, 0)
    
    # xs = resolutions_x[mask]
    # ys = resolutions_y[mask]
    # for i in range(resolution):
    #     for j in range(resolution):
    #         if mask[i, j] == True:
    #             if mask[i+1, j] == True and mask[i+1, j+1] == True and mask[i+1, j-1] == True:
    #                 mask[i, j] = False
    #             if mask[i-1, j] == True and mask[i-1, j+1] == True and mask[i-1, j-1] == True:
    #                 mask[i, j] = False
    #             if mask[i, j+1] == True and mask[i+1, j+1] == True and mask[i-1, j+1] == True:
    #                 mask[i, j] = False
    #             if mask[i, j-1] == True and mask[i+1, j-1] == True and mask[i-1, j-1] == True:
    #                 mask[i, j] = False
    original_mask = mask.copy()
    vct.check_mask(mask)
    # make these points Manhattan traverable
    dup_mask = mask.copy()
    lenmask = mask.sum()
    
    # del_x = m_idcs_x[nearest]
    # del_y = m_idcs_y[nearest]
    bfs_queue = np.zeros((lenmask, 2), dtype=np.int64)
    bfs_front = 0
    bfs_back = 0
    # dup_mask[del_x,del_y] = False
    
    connection = []
    rev_edges = []
    def check9(idx, idy):
        b1 = idx + 1 < resolution and dup_mask[idx + 1, idy] == True
        b2 = idy + 1 < resolution and dup_mask[idx, idy + 1] == True
        b3 = idx - 1 >= 0 and dup_mask[idx - 1, idy] == True
        b4 = idy - 1 >= 0 and dup_mask[idx, idy - 1] == True
        return b1 or b2 or b3 or b4
    while bfs_front < lenmask:
        # start
        bfs_back += 1
        connection.append(bfs_front)
        xs = resolutions_x[dup_mask]
        sel_xmax = xs.argmax()
        m_idcs_x = idcs_x[dup_mask]
        m_idcs_y = idcs_y[dup_mask]
        idcs_xmax = m_idcs_x[sel_xmax]
        idcs_ymax = m_idcs_y[sel_xmax]
        bfs_queue[bfs_front, :] = idcs_xmax, idcs_ymax
        tidcsx, tidcsy = idcs_xmax, idcs_ymax
        dup_mask[tidcsx, tidcsy] = False

        offset = 1
        # while idcs_xmax + offset < resolution and original_mask[idcs_xmax + offset, idcs_ymax] == True:
        #     offset += 1
        # if idcs_xmax + offset < resolution and png[idcs_xmax + offset, idcs_ymax] == 0:
        #     rev_edge = True
        # else:
        #     rev_edge = False
        while original_mask[idcs_xmax + offset, idcs_ymax] == True:
            offset += 1
        if png[idcs_xmax + offset, idcs_ymax] == 0:
            rev_edge = True
        else:
            rev_edge = False
        rev_edges.append(rev_edge)

        if tidcsx + 1 < resolution and dup_mask[tidcsx + 1, tidcsy] == True:
            bfs_queue[bfs_back, :] = tidcsx + 1, tidcsy
            dup_mask[tidcsx + 1, tidcsy] = False
            bfs_back += 1
        if tidcsy + 1 < resolution and dup_mask[tidcsx, tidcsy + 1] == True:
            bfs_queue[bfs_back, :] = tidcsx, tidcsy + 1
            dup_mask[tidcsx, tidcsy + 1] = False
            bfs_back += 1
        if tidcsx - 1 >= 0 and dup_mask[tidcsx - 1, tidcsy] == True:
            bfs_queue[bfs_back, :] = tidcsx - 1, tidcsy
            dup_mask[tidcsx - 1, tidcsy] = False
            bfs_back += 1
        if tidcsy - 1 >= 0 and dup_mask[tidcsx, tidcsy - 1] == True:
            bfs_queue[bfs_back, :] = tidcsx, tidcsy - 1
            dup_mask[tidcsx, tidcsy - 1] = False
            bfs_back += 1
        for cki in range(bfs_front + 1, bfs_back):
            if check9(*bfs_queue[cki]):
                tvar = np.zeros_like(bfs_queue[cki])
                tvar[:] = bfs_queue[cki, :]
                bfs_queue[cki, :] = bfs_queue[bfs_back - 1, :]
                bfs_queue[bfs_back - 1, :] = tvar[:]
                break
        bfs_front = bfs_back - 1
        while bfs_front < bfs_back:
            tidcsx, tidcsy = bfs_queue[bfs_front]
            if tidcsx + 1 < resolution and dup_mask[tidcsx + 1, tidcsy] == True:
                bfs_queue[bfs_back, :] = tidcsx + 1, tidcsy
                dup_mask[tidcsx + 1, tidcsy] = False
                bfs_back += 1
            if tidcsy + 1 < resolution and dup_mask[tidcsx, tidcsy + 1] == True:
                bfs_queue[bfs_back, :] = tidcsx, tidcsy + 1
                dup_mask[tidcsx, tidcsy + 1] = False
                bfs_back += 1
            if tidcsx - 1 >= 0 and dup_mask[tidcsx - 1, tidcsy] == True:
                bfs_queue[bfs_back, :] = tidcsx - 1, tidcsy
                dup_mask[tidcsx - 1, tidcsy] = False
                bfs_back += 1
            if tidcsy - 1 >= 0 and dup_mask[tidcsx, tidcsy - 1] == True:
                bfs_queue[bfs_back, :] = tidcsx, tidcsy - 1
                dup_mask[tidcsx, tidcsy - 1] = False
                bfs_back += 1
            bfs_front += 1
    connection.append(bfs_front)
    # rev_edges[0] = False
    for i in range(len(connection) - 1):
        if rev_edges[i]:
            bfs_queue[connection[i]:connection[i+1]] = bfs_queue[connection[i]:connection[i+1]][::-1]
    # import pdb; pdb.set_trace()
    ordered_vertices = bfs_queue.astype(np.float) / resolution
    normals = get_normal(ordered_vertices, connection)
    curv = get_curvature(normals, connection)

    # import pdb; pdb.set_trace()
    xys = np.expand_dims(ordered_vertices, 1)
    corners_sw = np.load(fcorners)
    if corners_sw.shape[0] != 0:
        cornersX = np.zeros_like(corners_sw)
        cornersX[:, 0], cornersX[:, 1] = corners_sw[:, 1], corners_sw[:, 0] 
        corners = cornersX
        sel_corner = (corners[:, 0] != 0.) & (corners[:, 1] != 0.)
        corners = corners[sel_corner]
        distance = np.linalg.norm(xys - np.expand_dims(corners, 0), axis=2)
        nearest_list = []
        for ci in range(len(connection)-1):
            nearest_list.append(distance[connection[ci]:connection[ci+1]].argmin(0)+connection[ci])
        nearest, corners = select_nearest(nearest_list, connection, ordered_vertices, corners, curv)
        # nearest = distance.argmin(0)
        # import pdb; pdb.set_trace()
        # return Combine_Rendering(ordered_vertices, corners, nearest_list, normals, curv, connection, resolution)
        return Combine_Rendering(ordered_vertices, corners[np.argsort(nearest)], np.sort(nearest), normals, curv, connection, resolution)
    else:
        nearest = np.array([connection[-1]], dtype=np.int64)
        return Combine_Rendering(ordered_vertices, np.array([]), nearest, normals, curv, connection, resolution)
    # Estimation of the normal and curvature
    
    
    # for c in corners_sw:
    #     ic = (c*1024).astype(np.int64)
    #     cv2.circle(output, ic, 3, 255)
    lv = len(ordered_vertices)
    # import pdb; pdb.set_trace()
    # corners = corners[np.argsort(nearest)]
    nearest = np.sort(nearest)
    # q = 7
    # ov = (ordered_vertices[nearest[q]]*resolution).astype(np.int64)
    # cv2.circle(output, [ov[1], ov[0]], 5, 255, 1)
    # ov = (corners[q]*resolution).astype(np.int64)
    # cv2.circle(output, [ov[1], ov[0]], 8, 255, 1)
    print(nearest)
    print(connection)
    # import pdb; pdb.set_trace()
    # output[mask] = 255
    for i in range(1400,1430):
        ov = (ordered_vertices[i]*resolution).astype(np.int64)
        # no = (normals[i]*50).astype(np.int64)
        # cv2.line(output, ov, ov+no, 255, 1)
        # cv2.circle(output, ov, int(curv[i]*5), 255, 1)
        cv2.circle(output, [ov[1], ov[0]], 2, 255, 1)
    # import pdb; pdb.set_trace()


    # pdb.set_trace()
    # for (x, y) in bfs_queue:
    #     # import pdb; pdb.set_trace()
    #     output[x, y] = 255
    #     cv2.circle(output, [x, y], 3, 255, 3)

    # for ne in nearest:
    #     cv2.circle(output, bfs_queue[ne], 10, 255, 2)
    cv2.imwrite('edge.png', output)
    cv2.imwrite('pgs.png', png)
    exit()

def select_nearest(nearest_list, connection, ordered_vertices, corners, curvature):
    # import pdb; pdb.set_trace()
    nearest = np.zeros_like(nearest_list[0])
    n_mask = np.zeros(nearest.shape, dtype=np.bool)
    for cori in range(len(corners)):
        sel_c = corners[cori]
        min_dis = 1000.0
        max_curv = 0.
        for coni in range(len(connection) - 1):
            id_nv = nearest_list[coni][cori]
            sel_v = ordered_vertices[id_nv]
            dis_cv = np.linalg.norm(sel_c-sel_v)
            if dis_cv > 0.2:
                continue
            elif dis_cv < min_dis:
                min_dis = dis_cv
                nearest[cori] = id_nv
                n_mask[cori] = True
                max_curv = curvature[id_nv]
            elif abs(min_dis - dis_cv) < 0.05 and curvature[id_nv] - max_curv > 0.25:
                min_dis = dis_cv
                nearest[cori] = id_nv
                n_mask[cori] = True
                max_curv = curvature[id_nv]
    return nearest[n_mask], corners[n_mask]

# using poisson distribution to decide the interpolation coefficient of Dual Contouring
def Poisson_distribution(lamb, n):
    po = np.zeros((n, ))
    facto = 1
    for k in range(1, n+1):
        facto *= k
        po[k-1] = np.exp(-lamb)*np.power(lamb, k)/facto
    return po

def fit_seq_normal(seq):
    norm = np.zeros(2)
    for i in range(5):
        dir = seq[20-i] - seq[i]
        norm += dir
    x, y = norm
    mag = np.math.sqrt(x*x+y*y)
    norm[0] = y/mag
    norm[1] = -x/mag
    return norm

def fit_seq_curv(seq):
    vx, vy = seq[10]
    lxy = math.sqrt(vx*vx+vy*vy)
    curv = 0
    for i in range(1, 6):
        vtx, vty = seq[10-i]
        cost = (vtx*vx+vty*vy)/math.sqrt(vtx*vtx+vty*vty)/lxy
        curv += abs(math.acos(np.clip(cost, -1, 1)))
        vtx, vty = seq[10+i]
        cost = (vtx*vx+vty*vy)/math.sqrt(vtx*vtx+vty*vty)/lxy
        curv += abs(math.acos(np.clip(cost, -1, 1)))
    return curv

def reselect_nearest(nearest, curv, connection, resolution):
    reselect_radius = int(resolution/1024.*40)
    connection_i = 0
    for i in range(len(nearest)):
        n_id = nearest[i]
        temp_curv = 0
        while n_id >= connection[connection_i+1]:
            connection_i += 1
        for _ in range(reselect_radius):
            n_id -= 1
            if n_id < connection[connection_i]:
                n_id = connection[connection_i+1] - 1
        for _ in range(reselect_radius*2+1):
            if curv[n_id] > temp_curv:
                temp_curv = curv[n_id]
                nearest[i] = n_id
            n_id += 1
            if n_id >= connection[connection_i+1]:
                n_id = connection[connection_i]

def dual_contouring(nearest, normals, ordered_vertices, connection, corners, resolution, coe):
    n_len = int(resolution/1024.*coe)
    lambda_poi = n_len
    pdis = Poisson_distribution(lambda_poi, lambda_poi)
    pdis /= pdis.sum()
    pdis = np.expand_dims(pdis,1).repeat(2,1)
    connection_i = 0
    id_seq = np.zeros((n_len, ), dtype=np.int64)
    new_corners = np.zeros_like(corners)
    for i in range(len(nearest)):
        n_id = nearest[i]
        while n_id >= connection[connection_i+1]:
            connection_i += 1
        for j in range(n_len):
            n_id -= 1
            if n_id < connection[connection_i]:
                n_id = connection[connection_i+1] - 1
            id_seq[j] = n_id
        new_normal_left = (normals[id_seq] * pdis).sum(axis=0)
        mag_n = np.linalg.norm(new_normal_left)
        if abs(mag_n) < 1e-6:
            new_corners[i] = corners[i]
            continue
        new_normal_left /= mag_n
        tangent_left = np.array([new_normal_left[1], -new_normal_left[0]])
        vert_left = (ordered_vertices[id_seq] * pdis).sum(axis=0)
        n_id = nearest[i]
        for j in range(n_len):
            n_id += 1
            if n_id >= connection[connection_i+1]:
                n_id = connection[connection_i]
            id_seq[j] = n_id
        new_normal_right = (normals[id_seq] * pdis).sum(axis=0)
        mag_n = np.linalg.norm(new_normal_right)
        if abs(mag_n) < 1e-6:
            new_corners[i] = corners[i]
            continue
        new_normal_right /= mag_n
        tangent_right = np.array([new_normal_right[1], -new_normal_right[0]])
        vert_right = (ordered_vertices[id_seq] * pdis).sum(axis=0)

        dvert = vert_right - vert_left
        # dtangent = tangent_left - tangent_right

        a1, b1, c1 = tangent_left[0], -tangent_right[0], dvert[0]
        a2, b2, c2 = tangent_left[1], -tangent_right[1], dvert[1]
        
        if abs(a1*b2 - a2*b1) < 1e-6:
            new_corners[i] = corners[i]
            continue
        t = (c1*b2 - c2*b1) / (a1*b2 - a2*b1)
        new_corners[i] = t*tangent_left + vert_left
        if np.linalg.norm(new_corners[i] - corners[i]) > 0.2:
            new_corners[i] = corners[i]

    corners[:] = new_corners[:]
                

# smoothing estimation of the surface normal of series of ordered points
def get_normal(ordered_vertices, connection):
    connection_i = 0
    normals = np.zeros_like(ordered_vertices)
    temp_seq = np.zeros((21, 2))
    offset = 0
    while connection_i < connection.__len__() - 1:
        vset = ordered_vertices[connection[connection_i]:connection[connection_i+1]]
        lvs = len(vset)
        sample_radius = min(10, lvs//2)
        for i in range(lvs):
            left = (i - sample_radius) % lvs
            for j in range(2*sample_radius+1):
                temp_seq[j, :] = vset[(left+j)%lvs, :]
            normals[offset+i, :] = fit_seq_normal(temp_seq)
        offset += lvs
        connection_i += 1
    return normals

def get_curvature(normals, connection):
    connection_i = 0
    curvature = np.zeros((normals.shape[0], ))
    temp_seq = np.zeros((21, 2))
    offset = 0
    while connection_i < connection.__len__() - 1:
        nset = normals[connection[connection_i]:connection[connection_i+1]]
        lvs = len(nset)
        sample_radius = min(10, lvs//2)
        for i in range(lvs):
            left = (i - sample_radius) % lvs
            for j in range(2*sample_radius+1):
                temp_seq[j, :] = nset[(left+j)%lvs, :]
            curvature[offset+i] = fit_seq_curv(temp_seq)
        offset += lvs
        connection_i += 1
    return curvature




def Vectorize_Image(f_img, f_corner, output_svg, resolution = 1024):
    paths_svg = Edge_Partition(f_img, f_corner, resolution)
    # print(paths_svg)
    with open(output_svg, 'w') as f:
        f.write(paths_svg)
    return paths_svg
    

if __name__ == '__main__':
    with torch.no_grad():
        # print(Edge_Partition('Datasets/Alphabet9/0.png', 'Datasets/Alphabet9/corners/0corner.npy', 1024))
        Vectorize_Image('results/doubleif13sine24reluganC250/197.png', 'results/doubleif13sine24reluganC250/197c.npy')