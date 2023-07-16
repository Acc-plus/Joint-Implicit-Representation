import cv2
import os
import numpy as np
import svgpathtools
from svgpathtools import svg2paths2
from svgpathtools import svg2paths, wsvg
from svgpathtools.path import Path, Line, QuadraticBezier, CubicBezier

class ImgCollection():
    def __init__(self):
        self.col = 13
        self.imgs = []
        self.extra_y = []
        self.resolution = 256

    def add(self, img):
        self.imgs.append(img)

    def render_img(self, output_file):
        n_img = len(self.imgs)
        col_img = self.col
        row_img = (n_img + col_img - 1) // col_img
        if row_img == 1:
            col_img = n_img
        output = np.ones((row_img*self.resolution, col_img*self.resolution), dtype=np.uint8)*255
        for i in range(n_img):
            path = self.imgs[i]
            tim = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            tim = cv2.resize(tim, (self.resolution, self.resolution))
            output[(i//col_img)*self.resolution:(i//col_img+1)*self.resolution, ((i%col_img)*self.resolution):(i%col_img+1)*self.resolution] = tim
        cv2.imwrite(output_file, output)

class SVGCollection():
    def __init__(self):
        self.col = 5
        self.imgs = []
        self.offsetx = 100
        self.offsety = 110j
        self.scale = 0.2
        
    def add(self, img):
        self.imgs.append(img)

    def render_img(self, output_file):
        n_img = len(self.imgs)
        pathlist = []
        attrlist = []
        svattrs = []
        count_path = 0
        # pathlist.append(Path())
        sel_path = 2
        ipath = 0
        for i in range(n_img):
            rid = i // self.col
            cid = i % self.col
            # pathlist.append(Path())
            svgp = self.imgs[i]
            paths, attributes, svg_attributes = svg2paths2(svgp)
            # import pdb; pdb.set_trace()
            # for j in range(len(paths[0])):
            #     pathlist[0].insert(count_path, transform(paths[0][j], self.scale, self.offsetx*cid+self.offsety*rid))
            #     count_path += 1
            for lp in range(len(paths)):
            # for lp in range(1, len(paths)-1):
                pathlist.append(Path())
                for j in range(len(paths[lp])):
                    pathlist[ipath].insert(j, transform(paths[lp][j], self.scale, self.offsetx*cid+self.offsety*rid))
                count_path += 1
                attrlist.append(attributes[lp])
                # print(len(paths))
                    # import pdb; pdb.set_trace()
                    # svattrs.append(svg_attributes[j])
                ipath += 1
            # import pdb; pdb.set_trace()
        svg_attributes['height'] = 2048
        # import pdb; pdb.set_trace()
        wsvg(pathlist, attributes=attrlist, svg_attributes=svg_attributes, filename=output_file)
        # wsvg(pathlist, attributes=attributes, svg_attributes=svg_attributes, filename=output_file)

def transform(cmd, scale, translate):
    # import pdb; pdb.set_trace()
    if cmd.__class__ == Line:
        return Line(cmd.start*scale+translate, cmd.end*scale+translate)
    elif cmd.__class__ == QuadraticBezier:
        return QuadraticBezier(cmd.start*scale+translate, cmd.control*scale+translate, cmd.end*scale+translate)
    elif cmd.__class__ == CubicBezier:
        return CubicBezier(cmd.start*scale+translate, cmd.control1*scale+translate, cmd.control2*scale+translate, cmd.end*scale+translate)

def get_svg(glyph, font_id):
    return f'./svg/{glyph}/{font_id}.svg'

def get_svg2(folder, glyph, style_id):
    return f'./demo/{folder}/{glyph_map[glyph]}_{style_id}.svg'

def get_svg_gt(glyph, font_id):
    return f'/mnt/data1/cjh/dvfsvgs/{glyph}_Test{glyph_map[glyph]}/{font_id}.svg'

def get_svg_dvf(glyph, font_id):
    return f'/mnt/data1/cjh/deepvecfont/experiments/dvfc_main_model/results/{font_id:04d}/svgs_refined/syn_{glyph_map[glyph]:02d}.svg'

def get_svg_im2vec(glyph, font_id):
    return f'/mnt/data1/cjh/Im2Vec/demo/{font_id}/{glyph_map[glyph]}.svg'

def get_svg_multi_imp(glyph, font_id):
    return f'/mnt/data1/cjh/multi_implicit_fonts/logs/cjhv1/{font_id:04d}/{glyph_map[glyph]:02d}_128.svg'

def get_png_attr2font(glyph, font_id):
    return f'/mnt/data1/cjh/Attr2Font/experiments/cjhdvfid/results/{font_id:04d}/{glyph_map[glyph]:02d}_ref_0003.png'

extra_y = {
    'm': 100j
}

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

if __name__ == '__main__':
# 15 83 95 208 for demo
# 1076, 420, 1078, 693, 118
    svcol = SVGCollection()
    interval = 26
    svcol.col = 26
    svcol.scale = 1/interval
    svcol.offsetx = 512/interval * 1
    svcol.offsety = 512j/interval*1.5
    # To_Render = 'ICCVFONTRECON'
    # To_Render = 'INTERPOLATION'
    # To_Render = 'ICCVFONTRECON'
    # To_Render = 'RECON'
    # To_Render = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    To_Render = 'abcdefghijklmnopqrstuvwxyz'
    # svcol.scale = 0.5/interval
    # svcol.offsetx = 512/interval
    # svcol.offsety = 512/interval*1.5
    # for id in [1076, 420, 1078, 693, 118]:
    folder = '0_1076'
    # for id in [0, 1, 2, 3, 11, 12, 15, 17, 19, 20]:
    # for id in [24, 25, 30, 31, 32, 33, 34, 35, 43, 59]:
    for folder in ['0_1076', '1076_999', '999_1117', '1117_1103']:
        for id in range(5):
            for glyph in To_Render:
                # svgp = get_svg_gt(glyph, 976)
                # svgp = get_svg_multi_imp(glyph, 976)
                # svgp = get_svg_dvf(glyph, 976)
                # svgp = get_svg(glyph, id)
                svgp = get_svg2(folder, glyph, id)
                svcol.add(svgp)
                # svgp = get_svg_im2vec(glyph, 976)


    # import pdb; pdb.set_trace()
    svcol.render_img('./suppl/interpo2.svg')
    exit()
    
    imcol = ImgCollection()
    imcol.col = 26
    # To_Render = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # # To_Render = 'abcdefghijklmnopqrstuvwxyz'
    for glyph in To_Render:
        # imcol.add(get_png_attr2font(glyph, 208))
        imcol.add(f'./renders/{glyph}/{208}.png')
    imcol.render_img('./zsvgs/ncf.png')
    # imcol.render_img('./lower.png')