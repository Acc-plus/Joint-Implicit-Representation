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



data_PATH = 'dataset/test_all.pkl'
f_dataset = 'dataset/dvfdata'

data = None
for i in range(52):
    data = render_corner_field(f_dataset, i, f'{rev_gm[i]}_Test', data_PATH = data_PATH, trainset=data)
#     data = render_svgs(f_dataset, 8035, i, f'{rev_gm[i]}_Test', data_PATH = data_PATH1, trainset=data)


