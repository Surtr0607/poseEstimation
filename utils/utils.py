import glob
import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import torch.utils.data
import cv2
from IPython import display
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment as lsa
from scipy.spatial.distance import cdist, cosine

'''
    utils package that used for image preprocessing
    including methods about background subtraction, sampling

'''


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

def draw_bodypose_front(filename, candidate, subset):
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    canvas = np.zeros((500,500,3), np.uint8)
    # 0,1 represent shoulder, 2 left arm, 3 left hand, 4 right arm
    # , 5 right hand, 6 left breast to knee, 7 left knee, 8 left leg, 9 right breast to knee, 10 right knee
    # 11 right leg, 12 throat, 14,16 to ear
    for i in range(17):
        # if i in [7]:
        #     continue
        if i in [0, 1,15,13,12,14,16,7,10,8,11]:
            for n in range(len(subset)):
                index = subset[n][np.array(limbSeq[i]) - 1]

                if -1 in index:
                    continue
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]

                cv2.line(canvas, (int(Y[0]), int(X[0])), (int(Y[1]), int(X[1])), (0,0,255), 1, 1)
    return canvas

def draw_bodypose_side(filename, candidate, subset):
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    canvas = np.zeros((500, 500,3), np.uint8)
    # 0,1 represent shoulder, 2 left arm, 3 left hand, 4 right arm
    # , 5 right hand, 6 left breast to knee, 7 left knee, 8 left leg, 9 right breast to knee, 10 right knee
    # 11 right leg, 12 throat, 14,16 to ear
    for i in range(17):
        # if i in [7]:
        #     continue
        if i in [0, 1,15,13,12,14,16, 11, 10,8]:
            for n in range(len(subset)):
                index = subset[n][np.array(limbSeq[i]) - 1]

                if -1 in index:
                    continue
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]

                cv2.line(canvas, (int(Y[0]), int(X[0])), (int(Y[1]), int(X[1])), (0,0,255), 1, 1)

    return canvas





def compare_visualize(img_a, img_b):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 8))
    ax1.imshow(img_a, cmap='gray_r')
    ax1.set_title('Original Skeleton')
    ax2.imshow(img_b, cmap='gray_r')
    ax2.set_title('Destination Skeleton')
    s1PointsList = np.array(img_a)
    s2PointsList = np.array(img_b)
    t1 = bdry_extract(cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY), s1PointsList)
    t2 = bdry_extract(cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY), s2PointsList)
    ax3.quiver(s1PointsList[:, 1], s1PointsList[:, 0], np.sin(t1), np.cos(t1), color='b')
    ax3.invert_yaxis()
    ax3.set_title(str(len(s1PointsList)) + '  samples on original skeleton')
    ax4.quiver(s2PointsList[:, 1], s2PointsList[:, 0], np.sin(t2), np.cos(t2), color='r')
    ax4.invert_yaxis()
    ax4.set_title(str(len(s2PointsList)) + '  samples on destination skeleton')
    plt.tight_layout()
    plt.show()


def bdry_extract(skeleton, pointList):
    t = np.zeros(len(pointList))
    G2, G1 = np.gradient(skeleton)
    for i, point in enumerate(pointList):
        t[i] = np.arctan2(G2[point[0], point[1]], G1[point[0], point[1]]) + np.pi / 2
    return t


def color2gray(img):
    minRange = 0
    maxRange = 1
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
        # Convert matrix to grayscale with the defined range
    minImg = np.min(img)
    maxImg = np.max(img)
    return (img - minImg) * (maxRange - minRange) / (maxImg - minImg) + minRange



def visualization(img, sample_points, index):
    """
    """
    histogram_feature = Shape_Context(sample_points)
    for i in range(len(index)):
        for point in sample_points:
            cv2.circle(img, (point[1], point[0]), 2, [0, 255, 0], 4)
        cv2.circle(img, (sample_points[i][1], sample_points[i][0]), 4, [255, 0, 0], 6)
        plt.subplot(2, len(index), i + 1)
        plt.axis('off')
        plt.imshow(img)

        plt.subplot(2, len(index), i + 1 + len(index))
        plt.axis('off')

        plt.imshow(histogram_feature[index[i]].astype(np.uint8))
    plt.show()






def Shape_Context(points):
        nbins_r = 5
        nbins_theta = 12
        r_inner = 0.1250
        r_outer = 2.0
        len_point = len(points)
        r_array = cdist(points, points)
        am = r_array.argmax()
        max_points = [am // len_point, am % len_point]
        # normalize
        r_array_normalized = r_array / r_array.mean()
        # create log space for shape context
        r_bin_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)
        r_array_q = np.zeros((len_point, len_point), dtype=int)
        # summing occurences in each log space intervals
        # logspace = [0.1250, 0.2500, 0.5000, 1.0000, 2.0000]
        for m in range(nbins_r):
            r_array_q += (r_array_normalized < r_bin_edges[m])

        fz = r_array_q > 0

        # getting angles in radians
        theta_array = cdist(points, points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
        norm_angle = theta_array[max_points[0],max_points[1]]
        # making angles matrix rotation invariant
        theta_array = (theta_array - norm_angle * (np.ones((len_point, len_point)) - np.identity(len_point)))
        # removing all very small values because of float operation
        theta_array[np.abs(theta_array) < 1e-7] = 0

        # 2Pi shifted because we need angels in [0,2Pi]
        theta_array = theta_array + 2 * math.pi * (theta_array < 0)
        # Simple Quantization
        theta_array_quantized = (1 + np.floor(theta_array / (2 * math.pi / nbins_theta))).astype(int)

        # building point descriptor based on angle and distance
        nbins = nbins_theta * nbins_r
        descriptor = np.zeros((len_point, nbins))
        for i in range(len_point):
            sn = np.zeros((nbins_r, nbins_theta))
            for j in range(len_point):
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_quantized[i, j] - 1] += 1
            descriptor[i] = sn.reshape(nbins)

        return descriptor


def split_test_images(paths, reference):
        side_images, front_images = [], []
        for path in paths:
            ref = reference[os.path.basename(path)]
            if ref.endswith("s.jpg"):
                side_images += [path]
            elif ref.endswith("f.jpg"):
                front_images += [path]
            else:
                raise RuntimeError("Unknown file ending")
        return side_images, front_images

def draw_bodypose(filename, candidate, subset):
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
    stickwidth = 4

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(filename, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = filename.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            filename = cv2.addWeighted(filename, 0.4, cur_canvas, 0.6, 0)
    return filename