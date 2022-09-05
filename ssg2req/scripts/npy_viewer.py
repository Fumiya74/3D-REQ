#import cv2
import numpy as np
import matplotlib.pyplot as plt

#bb = np.load('../../data/datasets/0988ea72-eb32-2e61-8344-99e2283c2728_0_bbox.npy', allow_pickle=True)
#print(bb)
#cap = np.load('../../data/datasets/0988ea72-eb32-2e61-8344-99e2283c2728_0_captions.npz')
#print(cap["arr_0"])
#cap = np.load('../../data/datasets/0988ea72-eb32-2e61-8344-99e2283c2728_0_question.npz')
#print(cap["arr_0"])
cap = np.load('test_pc.npz')
for c in cap['pc']:
    print(c)