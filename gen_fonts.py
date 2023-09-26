from utils.datagenerator import *

glyph_map = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15,
    'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25, 
    'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33,
    'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41,
    'q': 42, 'r': 43, 's': 44, 't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49,
    'y': 50, 'z': 51, 
}

rev_gm = dict()
for gl in glyph_map.keys():
    rev_gm[glyph_map[gl]] = gl



data_PATH2 = 'deepvecfont/data/vecfont_dataset/train/train_all.pkl'
data_PATH1 = '/mnt/data1/cjh/deepvecfont/data/vecfont_dataset_pkls/train/train_all.pkl'
f_dataset = 'dvfdata'

data = None
# for i in range(8, 52):
#     # data = render_corner_field(f_dataset, 8035, i, f'{rev_gm[i]}_Test', data_PATH = data_PATH1, trainset=data)
#     data = render_svgs(f_dataset, 8035, i, f'{rev_gm[i]}_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field('Datasets', 8035, 0, 'A_Train', data_PATH = data_PATH1)
# render_corner_field('Datasets', 8035, 0, 'A_Samp3k', data_PATH = data_PATH1)
# render_corner_field('Datasets', 8035, 0, 'A_Aug', data_PATH = data_PATH1)
# data = render_corner_field('Datasets', 8035, 1, 'B_Train', data_PATH = data_PATH1)
# render_corner_field('Datasets', 8035, 2, 'C_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 3, 'D_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 4, 'E_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 5, 'F_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 6, 'G_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 7, 'H_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 8, 'I_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 9, 'J_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 10, 'K_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 11, 'L_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 12, 'M_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 13, 'N_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 14, 'O_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 15, 'P_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 16, 'Q_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 17, 'R_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 18, 'S_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 19, 'T_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 20, 'U_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 21, 'V_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 22, 'W_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 23, 'X_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 24, 'Y_Train', data_PATH = data_PATH1, trainset=data)
# render_corner_field('Datasets', 8035, 25, 'Z_Train', data_PATH = data_PATH1, trainset=data)

data = render_corner_field(f_dataset, 8035, 0, 'A_Test', data_PATH = data_PATH1)
render_corner_field(f_dataset, 8035, 1, 'B_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 2, 'C_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 3, 'D_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 4, 'E_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 5, 'F_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 6, 'G_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 7, 'H_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 8, 'I_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 9, 'J_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 10, 'K_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 11, 'L_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 12, 'M_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 13, 'N_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 14, 'O_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 15, 'P_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 16, 'Q_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 17, 'R_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 18, 'S_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 19, 'T_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 20, 'U_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 21, 'V_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 22, 'W_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 23, 'X_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 24, 'Y_Test', data_PATH = data_PATH1, trainset=data)
render_corner_field(f_dataset, 8035, 25, 'Z_Test', data_PATH = data_PATH1, trainset=data)

render_corner_field('Datasets', 8035, 27, 'b_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 28, 'c_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 29, 'd_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 30, 'e_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 31, 'f_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 32, 'g_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 33, 'h_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 34, 'i_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 35, 'j_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 36, 'k_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 37, 'l_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 38, 'm_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 39, 'n_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 40, 'o_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 41, 'p_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 42, 'q_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 43, 'r_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 44, 's_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 45, 't_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 46, 'u_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 47, 'v_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 48, 'w_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 49, 'x_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 50, 'y_Train', data_PATH = data_PATH1)
render_corner_field('Datasets', 8035, 51, 'z_Train', data_PATH = data_PATH1)

