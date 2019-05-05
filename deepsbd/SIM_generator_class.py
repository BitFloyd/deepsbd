import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from sklearn.preprocessing import MinMaxScaler


class VideoLengthAssertionError(Exception):
    pass


class SlidingWindowLengthEvenError(Exception):
    pass


class SIMGenerator:

    def __init__(self,num_frames_per_sample):

        self.slidingWindowLength = num_frames_per_sample
        self.opticalFlowSIM = None
        self.sdimSIM = None
        self.featureMatchingSIM = None

        self.listOfFramesForCurrentShot = []

        self.farnBackParams = {'flow': None, 'pyr_scale': 0.75, 'levels': 7, 'winsize': 15, 'iterations': 3,
                               'poly_n': 7, 'poly_sigma': 1.2, 'flags': 0}
        self.frameResizeParams = {'fx': 1, 'fy':1}

        self.keypointDetector = self.getKeyPointDetector()

        self.debug_mode = False

    def prepFrames(self):

        self.prepOFFrames = []
        #PrepFramesForOpticalFlow
        for i in range(0, self.slidingWindowLength):
            f = self.listOfFrames[i]
            self.prepOFFrames.append(self.prepFrameForOpticalFlows(f))

        self.prepMCFrames = []
        for i in range(0,self.slidingWindowLength):
            f = self.prepOFFrames[i]
            self.prepMCFrames.append(self.prepFrameForMatchCheck(f))

        self.prepSDIMFrames = self.prepMCFrames

        return True

    def prepFrameForOpticalFlows(self, f):

        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = cv2.resize(f, (0, 0), fx=self.frameResizeParams['fx'], fy=self.frameResizeParams['fy'])

        return f

    def prepFrameForMatchCheck(self, f):

        f = cv2.GaussianBlur(f, ksize=(5, 5), sigmaX=1.0, sigmaY=0)

        return f

    def prepFrameForSDIMCheck(self, f):

        f = cv2.GaussianBlur(f, ksize=(5, 5), sigmaX=1.0, sigmaY=0)

        return f

    def getOpticalFlow(self, f1, f2):

        flow = cv2.calcOpticalFlowFarneback(prev=f1, next=f2, **self.farnBackParams)

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.medianBlur(mag, ksize=3)
        mag = np.median(mag)

        return mag

    def getSDIM(self, f1, f2):

        images_SDIM = 1 - compare_ssim(f1, f2, win_size=3 * self.farnBackParams['winsize'])

        return images_SDIM

    def getKeyPointDetector(self):

        keypointDetector = cv2.ORB_create()

        return keypointDetector

    def getMatchRatio(self, f1, f2):

        try:
            kp1, des1 = self.keypointDetector.detectAndCompute(f1, None)
            kp2, des2 = self.keypointDetector.detectAndCompute(f2, None)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good_matches = []
            image_diagonal = math.sqrt(f1.shape[0] ** 2 + f1.shape[1] ** 2)

            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    x1, y1 = kp1[m.queryIdx].pt
                    x2, y2 = kp2[m.trainIdx].pt
                    pt1 = np.array([x1, y1])
                    pt2 = np.array([x2, y2])
                    distance = np.linalg.norm(pt1 - pt2)
                    if (distance < 0.1 * image_diagonal):
                        good_matches.append([m])

            matches_ratio = (2.0 * len(good_matches)) / (len(kp1) + len(kp2))
            mismatch_ratio = 1 - matches_ratio

        except:
            mismatch_ratio = 1

        return mismatch_ratio

    def createOpticalFlowSIM(self):

        opticalFlowSIM = np.zeros((self.slidingWindowLength, self.slidingWindowLength))

        for i in range(0, self.slidingWindowLength):
            f2 = self.prepOFFrames[i]
            for j in range(0, self.slidingWindowLength):
                f1 = self.prepOFFrames[j]
                opticalFlowSIM[i, j] = self.getOpticalFlow(f1, f2)

        return opticalFlowSIM

    def createSdimSIM(self):

        sdimSIM = np.zeros((self.slidingWindowLength, self.slidingWindowLength))

        for i in range(0, self.slidingWindowLength):
            f2 = self.prepSDIMFrames[i]
            for j in range(0, self.slidingWindowLength):
                f1 = self.prepSDIMFrames[j]
                sdimSIM[i, j] = self.getSDIM(f1, f2)

        return sdimSIM

    def createFeatureMatchingSIM(self):

        featureMatchingSIM = np.zeros((self.slidingWindowLength, self.slidingWindowLength))

        for i in range(0, self.slidingWindowLength):
            f2 = self.prepMCFrames[i]
            for j in range(0, self.slidingWindowLength):
                f1 = self.prepMCFrames[j]
                featureMatchingSIM[i, j] = self.getMatchRatio(f1, f2)

        return featureMatchingSIM

    def createFullSIM(self,listOfFrames):

        self.listOfFrames = listOfFrames

        self.prepFrames()

        self.opticalFlowSIM = self.createOpticalFlowSIM()
        self.sdimSIM = self.createSdimSIM()
        self.featureMatchingSIM = self.createFeatureMatchingSIM()

        if (self.debug_mode):
            self.debugSaveSIM()

        SIM = cv2.merge((self.opticalFlowSIM,self.sdimSIM,self.featureMatchingSIM))

        return SIM

    def normalizeMatrix(self, matrix, min, max):

        scaler = MinMaxScaler(feature_range=(min, max))
        matrix = scaler.fit_transform(matrix)

        return matrix

    def debugSaveSIM(self):

        fig, axes = plt.subplots(3)

        axes[0].imshow(self.opticalFlowSIM, vmax=1.0, vmin=0.0, cmap='tab20b')
        axes[0].axis('off')
        axes[0].set_title('OPTICAL FLOW SIM')

        axes[1].imshow(self.sdimSIM, vmax=1.0, vmin=0.0, cmap='tab20b')
        axes[1].axis('off')
        axes[1].set_title('SDIM SIM')

        im = axes[2].imshow(self.featureMatchingSIM, vmax=1.0, vmin=0.0, cmap='tab20b')
        axes[2].axis('off')
        axes[2].set_title('FEATURE SIM')

        cb_ax = fig.add_axes([0.73, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cb_ax)

        filename = os.path.join(self.pathToFrameSIM,
                                str(self.frameNumber - self.slidingWindowLength / 2 - 1).zfill(6) + '_sim_matrix.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        return True



        return False

