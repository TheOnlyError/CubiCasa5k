import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from floortrans.plotting import discrete_cmap

files = os.walk('data/cubicasa5k')
j = 0
for f in files:
    j += 1
    if 'model.svg' not in f[2]:
        continue
    print(j, f[0])
    if j == 876 or j == 1810:
        continue

    path = f[0].replace('\\', '/')
    folder = path.split('/')[-1]

    fplan = cv2.imread(path + '/F1_original.png')
    fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
    height_org, width_org, _ = fplan.shape
    fplan = np.moveaxis(fplan, -1, 0)

    label = cv2.imread(path + '/label.png')
    label = np.moveaxis(label, -1, 0)
    label = torch.tensor(label)
    label = label.unsqueeze(0)
    label = torch.nn.functional.interpolate(label,
                                            size=(height_org, width_org),
                                            mode='nearest')
    label = label.squeeze(0)
    np_img = np.moveaxis(fplan, 0, -1)
    label_img = label.numpy()
    label_img = np.moveaxis(label_img, 0, -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(np_img)
    plt.imshow(label_img, alpha=0.6)

    plt.savefig('overlays/' + folder + '.jpg')
    plt.show()
    plt.close()
