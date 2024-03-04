import enum
from os import dup
import torch
import numpy as np
import math
from scipy.special import comb as nOk
from numpy import linalg, matrix
import cv2
import vct

def Bernstein(n, t, k):
    return t**(k)*(1-t)**(n-k)*nOk(n,k)
def Qbezier(ts):
    return matrix([[Bernstein(2,t,k) for k in range(3)] for t in ts])
def normalize(x, y):
    mag = math.sqrt(x*x + y*y)
    return np.array([x / mag, y / mag])

def chamfer_bezier(vqueue, bez_points):
    # not symmetric
    count_set2 = 0
    errs = np.zeros((vqueue.__len__(), ), dtype='float')
    for (i, p) in enumerate(vqueue):
        tempcmpp = bez_points[count_set2]
        err = np.linalg.norm(tempcmpp - p)
        count_set2 += 1
        while count_set2 < bez_points.__len__():
            tempcmpp = bez_points[count_set2]
            # Min Sqr Err
            temperr = np.linalg.norm(tempcmpp - p)
            if temperr < err:
                err = temperr
                count_set2 += 1
            else:
                break
        if count_set2 >= bez_points.__len__():
            count_set2 -= 1
        errs[i] = err
    return errs

def Vectorize_Edge(ordered_vertices, corner_st, corner_end):

    
    # Hyper parameters
    err_threshold_L_ave = 1. / 1024.
    err_threshold_L_max = 2. / 1024.
    err_threshold_Q_ave = 1. / 1024.
    err_threshold_Q_max = 2.5 / 1024.
    save_L_minlen = 0.1
    num_min_init = 10
    delta_diff = 1. / 1024.
    bisection_min_interval = 1. / 1024.
    err_threshold_inc = 1. / 1024
    smooth_min_angle = math.pi * 0.5 / 180.

    init_flag = True
    lenv = ordered_vertices.__len__()
    vqueue = []
    primitives = []
    parameters = []
    prim_verts = []
    st_direction = []
    end_direction = []
    controls = []
    previous_control = ordered_vertices[0]
    # the tangent direction of previous curve

    previous_direction = normalize(*(ordered_vertices[1] - ordered_vertices[0]))
    l_bisection = 0
    r_bisection = 0.1
    # -1: init
    # 0: L
    # 1: Q
    # 2: C Not used in ttf
    temp_prim = -1
    temp_param = np.array([0., 0., 0., 0.])
    
    # vert should not be modified
    vert_i = 0 # start from -1 + 1 = 0
    while vert_i < lenv:
        print(vert_i)
        # First, L gets higher priority, then smooth, then L, then Q
        vert = ordered_vertices[vert_i]
        if vqueue.__len__() == 0:
            vqueue.append(vert)
            previous_control = vert
            controls.append(vert)
        else:
            vqueue.append(vert)
            if vqueue.__len__() < num_min_init:
                pass
            # Analyse Error
            # ...
            elif temp_prim == -1:
                temp_prim = 0
                l_bisection = 0
                r_bisection = 0.1
            elif temp_prim == 0:
                x0, y0 = previous_control
                x1, y1 = vert
                a = y1 - y0
                b = x0 - x1
                c = x1*y0 - x0*y1
                proj = math.sqrt(a*a+b*b)
                accumulate_err = 0.
                max_err = 0.
                for v in vqueue:
                    err = np.abs(a*v[0] + b*v[1] + c) / proj
                    accumulate_err += err
                    max_err = max(max_err, err)
                # print(f'acc:{accumulate_err / vqueue.__len__()} max:{max_err}')
                temp_direction = normalize(*(vqueue[-1] - vqueue[0]))
                diff_angle = np.math.acos(np.dot(temp_direction, previous_direction))
                if diff_angle < smooth_min_angle and init_flag is False:
                    vert_i -= 1
                    temp_prim = 1
                    init_flag = False
                elif accumulate_err / vqueue.__len__() > err_threshold_L_ave or \
                    max_err > err_threshold_L_max:
                    if np.linalg.norm(vert - previous_control) > save_L_minlen:
                        vqueue.pop()
                        primitives.append(0) # Line
                        parameters.append(np.array(vqueue[-1]))
                        prim_verts.append(vqueue)
                        vqueue = []
                        # Recursive Solve
                        vert_i -= 1
                        # Reset
                        temp_prim = -1
                        previous_direction = normalize(x1 - x0, y1 - y0)
                        st_direction.append(np.array(previous_direction))
                        end_direction.append(np.array(previous_direction))
                        init_flag = False
                    else:
                        # maybe a Bezier Curve
                        vert_i -= 1
                        temp_prim = 1
                        init_flag = False
            elif temp_prim == 1:
                # vqueue.append(vert)
                sample_t = np.linspace(0, 1, vqueue.__len__() * 2)
                bezierM = Qbezier(sample_t)
                # print(f'{l_bisection} {r_bisection}')
                while(r_bisection - l_bisection > bisection_min_interval):
                    mid_bisection = (l_bisection + r_bisection) / 2
                    mid_ctrl = previous_control + previous_direction * mid_bisection
                    dif_mid_ctrl = previous_control + previous_direction * (mid_bisection + delta_diff)
                    ctrls = [previous_control, mid_ctrl, vert]
                    dif_ctrls = [previous_control, dif_mid_ctrl, vert]
                    bez_points = bezierM * ctrls
                    dif_bez_points = bezierM * dif_ctrls
                
                    errs = chamfer_bezier(vqueue, bez_points)
                    dif_errs = chamfer_bezier(vqueue, dif_bez_points)

                    if dif_errs.mean() < errs.mean():
                        l_bisection = mid_bisection
                    else:
                        r_bisection = mid_bisection
                    
                # Neighbourhood extension for next search
                l_bisection -= bisection_min_interval * 8
                r_bisection += bisection_min_interval * 8
                
                # print(f'{errs.mean()} {errs.max()} {len(primitives)}')
                if errs.mean() > err_threshold_Q_ave or \
                    errs.max() > err_threshold_Q_max:
                    vqueue.pop()
                    primitives.append(1)
                    parameters.append(np.concatenate([np.array(dif_mid_ctrl), np.array(vert)]))
                    prim_verts.append(vqueue)
                    vqueue = []
                    # Reset
                    vert_i -= 1
                    temp_prim = -1
                    st_direction.append(np.array(previous_direction))
                    previous_direction = normalize(*(vert - mid_ctrl))
                    end_direction.append(np.array(previous_direction))
        vert_i += 1
    
    # process the remain
    if temp_prim == 0:
        primitives.append(0) # Line
        parameters.append(np.array(vqueue[-1]))
        prim_verts.append(vqueue)
        previous_direction = normalize(*(vqueue[-1] - vqueue[0]))
        st_direction.append(np.array(previous_direction))
        end_direction.append(np.array(previous_direction))
    elif temp_prim == 1:
        primitives.append(1)
        parameters.append(np.concatenate([np.array(dif_mid_ctrl), np.array(vert)]))
        prim_verts.append(vqueue)
        st_direction.append(np.array(previous_direction))
        previous_direction = normalize(*(vert - mid_ctrl))
        end_direction.append(np.array(previous_direction))
    # import pdb; pdb.set_trace()
    front_count = 0
    back_count = primitives.__len__()
    # Corner Insertion
    for i, prim in enumerate(primitives):
        if prim == 0:
            temp_direction = normalize(*(parameters[i] - corner_st))
            diff_angle = np.math.acos(np.dot(temp_direction, st_direction[i]))
            if diff_angle < smooth_min_angle or np.linalg.norm(parameters[i][:2] - corner_st) > 0.05:
                break
        elif prim == 1:
            if np.linalg.norm(parameters[i][:2] - corner_st) > 0.05:
                controls[i] = corner_st
                break
        front_count += 1
    
    rev_prims = reversed(list(range(front_count, back_count)))
    for i in rev_prims:
        prim = primitives[i]
        if primitives[i] == 0:
            temp_direction = normalize(*(corner_end - controls[i]))
            diff_angle = np.math.acos(np.dot(temp_direction, end_direction[i]))
            if diff_angle < smooth_min_angle or back_count == 1:
                parameters[i] = corner_end
                break
        elif prim == 1:
            if np.linalg.norm(parameters[i][:2] - corner_end) > 0.05:
                parameters[i][2:] = corner_end[:]
                break
        back_count -= 1
    # if back_count == 0:
    #     back_count = 1
    #     primitives[0] = 0
    #     parameters[0] = corner_end

    if front_count >= back_count:
        # import pdb; pdb.set_trace()
        return [0], [corner_end]
    return primitives[front_count:back_count], parameters[front_count:back_count]

        
def Translate_Primitives(primitives, parameters, resolution = 1024):
    paths = []
    for prim, param in zip(primitives, parameters):
        param = param * resolution
        if prim == 0:
            paths.append(f"L {param[1]} {param[0]}")
        elif prim == 1:
            paths.append(f"Q {param[1]} {param[0]} {param[3]} {param[2]}")
    
    return ' '.join(paths) + '\n'

# 
def Combine_Rendering(ordered_vertices, corners, nearest, connection, rev_edges):
    print('vectorize start')
    resolution = 1024
    # insert cutting points to ordered vertices
    # ordered vertices represents recurrent vertices, which is a closed graph
    b_vert = np.ones((ordered_vertices.shape[0], ), np.bool)
    connection_i = 0
    temp_verts = []
    Primitives = []
    nearest_back = 0
    path_str = ''
    # assert connection.__len__() - 1 == 1
    nearest, ind_c = np.unique(nearest, return_index=True)
    corners = corners[ind_c]
    while connection_i < connection.__len__() - 1:
        
        vert_i = connection[connection_i]
        nearest_i = nearest_back
        nearest_front = nearest_back
        path_str += f'M {corners[nearest_i][1] * resolution} {corners[nearest_i][0] * resolution} '
        while vert_i != nearest[nearest_i]:
            vert_i += 1
        prev_nearest = nearest_i
        nearest_i += 1
        if nearest_i >= len(nearest) or nearest[nearest_i] >= connection[connection_i + 1]:
            nearest_i = nearest_front
            nearest_back = nearest_i + 1
        
        while b_vert[vert_i] == True:
            
            b_vert[vert_i] = False
            temp_verts.append(ordered_vertices[vert_i])
            vert_i += 1
            if vert_i >= connection[connection_i + 1]:
                vert_i = connection[connection_i]
            if vert_i == nearest[nearest_i]:
                cst = prev_nearest
                cend = nearest_i
                prev_nearest = nearest_i
                nearest_i += 1
                if nearest_i >= len(nearest) or nearest[nearest_i] >= connection[connection_i + 1]:
                    nearest_back = nearest_i + 1
                    nearest_i = nearest_front
                # primitives, parameters = Vectorize_Edge(ordered_vertices=temp_verts, 
                #     corner_st=corners[cst],
                #     corner_end=corners[cend])
                # import pdb; pdb.set_trace()
                primitives, parameters = vct.vectorize_edge(np.array(temp_verts), 
                    corners[cst],
                    corners[cend])
                # import pdb; pdb.set_trace()
                parameters = parameters.reshape(-1, 4)
                path_str += Translate_Primitives(primitives, parameters, resolution)
                temp_verts.clear()
            # temp_verts.append(ordered_vertices[vert_i])
        connection_i += 1

    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{resolution}px" height="{resolution}px" style="-ms-transform: rotate(360deg); -webkit-transform: rotate(360deg); transform: rotate(360deg);" preserveAspectRatio="xMidYMid meet" viewBox="0 0 {resolution} {resolution}">'
    f'<path d="{path_str}" stroke-width="1.0" fill="rgb(0, 0, 0)" opacity="1.0"></path></svg>')

    
device = torch.device('cuda:4')
def Edge_Partition(fpng, fcorner, resolution):
    output = np.zeros((resolution, resolution), dtype = np.uint8 )
    png = cv2.imread(fpng, cv2.IMREAD_GRAYSCALE) # GRAYSCALE uint8
    sdf = np.array(png)
    png[png > 128] = 255
    png[png <= 128] = 0
    png = png.astype(np.float32)
    png = torch.from_numpy(png).to(device)
    kernel = torch.ones((3,3), device=device)
    mask1 = png == 0
    mask2 = torch.conv2d(png.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=1) > 0
    mask = mask1 & mask2
    mask = mask.view(resolution, resolution).cpu().numpy()
    mask_count = mask.sum()
    resolutions = np.array([i/float(resolution) for i in range(resolution)], dtype=np.float32)
    resolutions_x = np.expand_dims(resolutions, 1).repeat(resolution, 1)
    resolutions_y = np.expand_dims(resolutions, 0).repeat(resolution, 0)
    idcs = np.array(range(resolution))
    idcs_x = np.expand_dims(idcs, 1).repeat(resolution, 1)
    idcs_y = np.expand_dims(idcs, 0).repeat(resolution, 0)
    
    # xs = resolutions_x[mask]
    # ys = resolutions_y[mask]
    

    dup_mask = mask.copy()
    lenmask = mask.sum()
    
    # del_x = m_idcs_x[nearest]
    # del_y = m_idcs_y[nearest]
    bfs_queue = np.zeros((mask_count, 2), dtype=np.int64)
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

        if png[idcs_xmax + 1, idcs_ymax] == 0:
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
            dup_mask[tidcsx - 1, tidcsy] = False
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
    for i in range(len(connection) - 1):
        if rev_edges[i]:
            bfs_queue[connection[i]:connection[i+1]] = bfs_queue[connection[i]:connection[i+1]][::-1]
    ordered_vertices = bfs_queue.astype(np.float) / resolution
    xys = np.expand_dims(ordered_vertices, 1)
    corners_sw = np.load(fcorner)
    # corners = corners_sw
    corners = np.zeros_like(corners_sw)
    corners[:, 0], corners[:, 1] = corners_sw[:, 1], corners_sw[:, 0] 
    sel_corner = (corners[:, 0] != 0.) & (corners[:, 1] != 0.)
    corners = corners[sel_corner]
    corners = np.expand_dims(corners, 0)
    distance = np.linalg.norm(xys - corners, axis=2)
    nearest = distance.argmin(0)

    corners = corners.squeeze(0)

    return Combine_Rendering(ordered_vertices, corners[np.argsort(nearest)], np.sort(nearest), connection, rev_edges)

    # for c in corners_sw:
    #     ic = (c*1024).astype(np.int64)
    #     cv2.circle(output, ic, 3, 255)
    for (x, y) in bfs_queue:
        output[x, y] = 255
        cv2.circle(output, [x, y], 3, 255, 3)

    for ne in nearest:
        cv2.circle(output, bfs_queue[ne], 10, 255, 2)
    cv2.imwrite('edge.png', output)
    
def Vectorize_Image(f_img, f_corner, output_svg, resolution = 1024):
    paths_svg = Edge_Partition(f_img, f_corner, resolution)
    print(paths_svg)
    with open(output_svg, 'w') as f:
        f.write(paths_svg)
    

if __name__ == '__main__':
    with torch.no_grad():
        # print(Edge_Partition('Datasets/Alphabet9/0.png', 'Datasets/Alphabet9/corners/0corner.npy', 1024))
        Vectorize_Image('results/doubleif13sine24reluganC250/197.png', 'results/doubleif13sine24reluganC250/197c.npy')