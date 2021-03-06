import sys
import re
import unpack
import numpy as np

import re

import numpy as np
import torch
import configparser
import os

config = configparser.ConfigParser()
config.read(os.path.join("configs", "sceneflow.config"))


def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()

    max_disparity = config.getint("CostVolume", "max_disparity")
    data = (data / data.max()) * max_disparity
    data = torch.from_numpy(data.copy())
    return data, scale
