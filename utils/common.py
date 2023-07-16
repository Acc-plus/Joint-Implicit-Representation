
import numpy as np

def gen_heatmap(x, y, mu_x, mu_y, sigma):
    return np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))


