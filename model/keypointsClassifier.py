import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from model.keypointsDetector import Body
from utils.utils import *
from scipy.spatial.distance import cosine, pdist
from scipy.stats import pearsonr

# the classifier will read all keypoints of a body and
# OCR method to skeleton matching
class KeypointsClassifier:
    def __init__(self, train_dir):
        dim = (500,500)
        self.body_estimation = Body('./body_pose_model.pth')
        # store all re-sample points from skeleton
        self.feature_points = {}
        front_train_dir = train_dir + '\\' + 'front_image'
        side_train_dir = train_dir + '\\' + 'side_image'


        front_train_images = os.listdir(front_train_dir)
        for train_image in front_train_images:
            img = cv2.imread(os.path.join(front_train_dir, train_image))
            img = cv2.resize(img, dim)
            candidate, subset = self.body_estimation(img)

            # obtain the virtual skeleton
            skeleton = draw_bodypose_front(train_image, candidate, subset)
            skeleton = cv2.cvtColor(skeleton, cv2.COLOR_RGB2GRAY)
            pointList = np.array(skeleton)
            prefix, suffix = train_image.split('p', 1)
            self.feature_points[prefix] = [Shape_Context(pointList)]

        side_train_images = os.listdir(side_train_dir)
        for train_image in side_train_images:
            img = cv2.imread(os.path.join(side_train_dir, train_image))
            img = cv2.resize(img, dim)
            candidate, subset = self.body_estimation(img)

            # obtain the virtual skeleton
            skeleton = draw_bodypose_side(train_image, candidate, subset)
            skeleton = cv2.cvtColor(skeleton, cv2.COLOR_RGB2GRAY)
            pointList = np.array(skeleton)
            prefix, suffix = train_image.split('p', 1)
            self.feature_points[prefix].append(Shape_Context(pointList))

    def getShapeContextDict(self):
        return self.feature_points

    def test(self, img_mapping):
        test_dir = r"C:\Users\34779\PycharmProjects\COMP6211CW\res\test"
        test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
        side_test_img, front_test_img = split_test_images(test_images, img_mapping)





        dim = (500,500)
        assert len(front_test_img)==len(side_test_img)
        n = len(front_test_img)
        res_label = []
        x = [i for i in range(n)]
        y1 = []
        y2 = []
        distance_list = {}
        for i in range(n):
            # for every test images pair
            front_img = cv2.imread(front_test_img[i])
            side_img = cv2.imread(side_test_img[i])

            distance_list[i] = []

            front_img = cv2.resize(front_img, dim)
            side_img = cv2.resize(side_img, dim)

            candidate, subset = self.body_estimation(front_img)
            target_front_skeleton = draw_bodypose_front(front_img, candidate, subset)
            target_front_skeleton = cv2.cvtColor(target_front_skeleton, cv2.COLOR_RGB2GRAY)
            front_pointList = Shape_Context(np.array(target_front_skeleton))

            candidate, subset = self.body_estimation(side_img)
            target_side_skeleton = draw_bodypose_side(side_img, candidate, subset)
            target_side_skeleton = cv2.cvtColor(target_side_skeleton, cv2.COLOR_RGB2GRAY)
            side_pointList = Shape_Context(np.array(target_side_skeleton))

            min_distance1, min_distance2 = 10000000000, 100000000000
            min_point1, min_point2 = '', ''

            for name in self.feature_points.keys():

                distance1 = cosine(self.feature_points[name][0].flatten(), front_pointList.flatten())
                distance2 = cosine(self.feature_points[name][1].flatten(), side_pointList.flatten())
                distance_list[i].append([name, [distance1, distance2]])
                if distance1 < min_distance1:
                    min_distance1 = distance1
                    min_point1 = name
                if distance2 < min_distance2:
                    min_distance2 = distance2
                    min_point2 = name
            res_label.append([min_point1, min_point2])
            y1.append(min_distance1)
            y2.append(min_distance2)

        l1 = plt.plot(x, y1, 'r--', label='front')
        l2 = plt.plot(x, y2, 'g--', label='side')
        plt.plot(x,y1,'ro-',x,y2,'g+-')
        plt.title('min_distance')
        plt.xlabel('sequence')
        plt.ylabel('cosine distance')
        plt.legend()
        plt.show()




        return res_label, distance_list












