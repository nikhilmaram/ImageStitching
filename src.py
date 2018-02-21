## Importing necessary packages

## numpy for matrix operations
import numpy as np
## openCV bindings
import cv2
## image convenience methods
import imutils
import matplotlib.gridspec as gridspec
import os
import sys

from homography import *

from natsort import natsorted

import matplotlib.pyplot as plt


class Stitcher:

    def __init__(self):
        pass
        # print("Sticher class object created")

    def visualiseImages(self,imageA,imageB):
        plt.subplot(1,2,1).imshow(imageA)
        plt.subplot(1,2,2).imshow(imageB)
        plt.show()

    def drawKeyPoints(self,image):
        sift = cv2.xfeatures2d.SIFT_create()

        gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)

        kps,features = sift.detectAndCompute(gray,None)

        kpImage = cv2.drawKeypoints(gray,keypoints=kps,outImage=None)

        plt.imshow(kpImage)
        plt.show()


    def stitch(self,images,ratio = 0.8, threshold = 5, showMatches = False):
        #print("In stitch function")
        ## get the images
        (imageA,imageB) = images

        ## Get the local invariant descriptors

        (kpsA,featuresA) = self.detectKeyPointsAndFeatures(imageA)
        (kpsB,featuresB) = self.detectKeyPointsAndFeatures(imageB)

        match = self.matchKeyPoints(kpsA,featuresA,kpsB,featuresB,ratio,threshold)
        ##match = self.matchKeyPointsWithImplementedRANSAC(kpsA, featuresA, kpsB, featuresB, ratio)

        ## Modified to avoid none condition
        if match is None:
            return imageA
            #return None

        ## Unpacking the returned tuples from matchKeyPoints function

        (matches,H,status) = match

        ## Now apply persepective warp on the images

        ## image.shape[1] is added since they represent the number of columns which is the width.
        ## We do a perspective projection of imageA and later add image B
        result = cv2.warpPerspective(imageA,H,(imageA.shape[1] + imageB.shape[1],imageA.shape[0]))

        ## Not sure why this is present - Ans. we are only warping imageA, then need to add image B
        result[0:imageB.shape[0],0:imageB.shape[1]] = imageB

        return result


    def detectKeyPointsAndFeatures(self,image):

        ## Converting the image to grey scale
        ##gray = cv2.cvtColor(image,cv2.COLOR_BG2AGRAY)

        ## Detect and extract features from image
        descriptor = cv2.xfeatures2d.SIFT_create()

        (kps,features) = descriptor.detectAndCompute(image,mask = None)
        ## Converting the keypoints into numpy array from keypoints
        kps = np.float32([kp.pt for kp in kps])

        return(kps,features)


    def matchKeyPoints(self,kpsA,featuresA,kpsB,featuresB,ratio,threshold):

        ## Compute the raw matches and intialise the list actual matches

        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA,featuresB,2)
        matches = []

        for match in rawMatches:
            if (len(match) == 2 and match[0].distance < match[1].distance * ratio):
                matches.append((match[0].trainIdx,match[0].queryIdx))

        if(len(matches) > 4):
            ## 4 points are needed for calculating the homography
            ptsA = np.float32([kpsA[i] for (_,i) in matches])
            ptsB = np.float32([kpsB[i] for (i,_) in matches])

            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             threshold)
            return (matches, H, status)

        return None

    def matchKeyPointsWithImplementedRANSAC(self, kpsA, featuresA, kpsB, featuresB, ratio, threshold = 0.7):

        ## Compute the raw matches and intialise the list actual matches
        matchingList = []
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for match in rawMatches:
            if (len(match) == 2 and match[0].distance < match[1].distance * ratio):
                matches.append((match[0].trainIdx, match[0].queryIdx))

        for match in matches:
            ## 4 points are needed for calculating the homography
            (x1,y1) = kpsA[match[1]]
            (x2,y2) = kpsB[match[0]]

            matchingList.append([x1, y1, x2, y2])

            correlations = np.matrix(matchingList)

            finalH, inliers = ransac(correlations, threshold)

            return (0,finalH,inliers)

        return None

    ## Removing the black area in the image
    def cropImageOnly(self,image):
        if image is None:
            return None
        ## Convert the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        ## Find threshold
        ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        ## Find contours for finding the original image and removing the black area
        _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        ## Get the first contour
        try:
            cnt = contours[0]
            ## Get the rectangle bounding that contour
            x, y, w, h = cv2.boundingRect(cnt)
            ## Crop the image only removing the black area
            cropImage= image[y:y+h,x:x+w]

            return cropImage
        except:
            return image

    ## Now stitch the images given the directory

    def stitchImagesInDirectory(self,dir):
        types = [".jpg",".JPG",".JPEG",".png"]
        files = natsorted(os.listdir(dir))

        ## Getting files of types required

        dirImages = [im for im in files if im.endswith(tuple(types))]

        print(os.path.join(dir,dirImages[0]))
        print(dirImages)
        try:
            stitchedImage = cv2.imread(os.path.join(dir,dirImages[0]))
            for im in dirImages:
                print("stitched the image ",im)
                image = cv2.imread(os.path.join(dir,im))
                stitchedImage = self.stitch((stitchedImage,image))
                stitchedImage = self.cropImageOnly(stitchedImage)
                tempDir = dir + '/stitched/'
                tempPath = os.path.join(tempDir,im)
                ##print(tempImage)
                ##stitchedImage = stitchedImage[:, :4000, :]
                cv2.imwrite(tempPath, stitchedImage)
        except :
            print("No Images in the directory")

        return stitchedImage

    def stitchImagesInDirectoryAlternately(self,dir):
        types = [".jpg", ".JPG", ".JPEG"]
        files = natsorted(os.listdir(dir))

        ## Getting files of types required

        dirImages = [im for im in files if im.endswith(tuple(types))]

        print(os.path.join(dir, dirImages[0]))
        print(dirImages)
        lenImages = len(dirImages)
        mid = int(lenImages/2)
        left = mid -1
        right = mid +1

        stitchedImageLeft = cv2.imread(os.path.join(dir, dirImages[mid]))
        while(left >=0 or right < lenImages):
        #while(left >= 0):
            if(left >= 0):
                print("left stitching")
                print (left)
                image = cv2.imread(os.path.join(dir, dirImages[left]))
                stitchedImageLeft = self.stitch((image,stitchedImageLeft))
                #stitchedImageLeft = self.cropImageOnly(stitchedImageLeft)
                tempDir = dir + '/stitched/'
                tempPath = os.path.join(tempDir, dirImages[left])
                ##print(tempImage)
                cv2.imwrite(tempPath, stitchedImageLeft)
                left = left -1
                stitchedImageRight = stitchedImageLeft
        #stitchedImageRight = cv2.imread(os.path.join(dir, dirImages[mid]))
        #while(right < lenImages):
            if(right < lenImages):
                print("right stitching")
                print(right)
                image = cv2.imread(os.path.join(dir, dirImages[right]))
                stitchedImageRight = self.stitch((stitchedImageRight,image))
                #stitchedImageRight = self.cropImageOnly(stitchedImageRight)
                tempDir = dir + '/stitched/'
                tempPath = os.path.join(tempDir, dirImages[right])
                ##print(tempImage)
                cv2.imwrite(tempPath, stitchedImageRight)
                right = right + 1
                stitchedImageLeft = stitchedImageRight

        stitchedImage = self.stitch((stitchedImageLeft,stitchedImageRight))
        stitchedImage = stitchedImage[:, :20000, :]
        return stitchedImage

    def stichImagesRecursively(self,dir):
        types = [".jpg", ".JPG", ".JPEG",".png"]
        files = natsorted(os.listdir(dir))

        ## Getting files of types required
        dirImages = [im for im in files if im.endswith(tuple(types))]

        #print(os.path.join(dir, dirImages[0]))
        #print(dirImages)
        lenImages = len(dirImages)

        left = 0
        right = lenImages -1

        stitchedImage = self.mergeSort(dir,dirImages,left,right)
        return stitchedImage

    def mergeSort(self,dir,dirImages,left,right):
        if(left == right):
            #print(left)
            return cv2.imread(os.path.join(dir, dirImages[left]))
        if(left < right):
            mid = int((left+right)/2)
            leftImage = self.mergeSort(dir,dirImages,left,mid)
            rightImage = self.mergeSort(dir,dirImages,mid+1,right)

            stitchedImage = self.merge((leftImage,rightImage),mid,dir,dirImages)

            tempDir = dir + '/stitched/'
            tempPath = os.path.join(tempDir, str(dirImages[left])+str(dirImages[right]))

            stitchedImage = self.cropImageOnly(stitchedImage)
            #print(stitchedImage.shape)
            #stitchedImage = stitchedImage[:,:5000,:]
            cv2.imwrite(tempPath, stitchedImage)

            return stitchedImage

    def merge(self,images,mid,dir,dirImages):

        leftImage,rightImage = images

        middleImage = self.stitch((cv2.imread(os.path.join(dir, dirImages[mid])),cv2.imread(os.path.join(dir, dirImages[mid+1]))))
        middleImage = self.cropImageOnly(middleImage)

        leftMost = self.stitch((leftImage,middleImage))
        leftMost = self.cropImageOnly(leftMost)

        rightMost = self.stitch((middleImage,rightImage))
        rightMost = self.cropImageOnly(rightMost)

        #leftMost = leftMost[:,:10000,:]
        #rightMost = rightMost[:,:10000,:]
        return self.stitch((leftMost,rightMost))



if __name__ == '__main__':

    st = Stitcher()
    #dir = sys.argv[1]

    dir = '/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Advance Topics in Computer Vision/HW1/Data/office/'
    ##dir = '/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Advance Topics in Computer Vision/HW1/Data/intersection/'

    stitchedImage = st.stitchImagesInDirectory(dir)

    #stitchedImage = st.stitchImagesInDirectoryAlternately(dir)
    ##stitchedImage = st.stichImagesRecursively(dir)
    stitchedImage = st.cropImageOnly(stitchedImage)

    #plt.imshow(stitchedImage)
    #plt.show()
    cv2.imwrite(os.path.join(dir,'myMosaic.jpg'),stitchedImage)




