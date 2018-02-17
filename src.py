## Importing necessary packages

## numpy for matrix operations
import numpy as np
## openCV bindings
import cv2
## image convenience methods
import imutils
import matplotlib.gridspec as gridspec


import matplotlib.pyplot as plt

print(cv2.__version__)

class Stitcher:

    def __init__(self):
        print("Sticher class object created")

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


    def stitch(self,images,ratio = 0.75, threshold = 4.0, showMatches = False):
        ## get the images
        (imageA,imageB) = images

        ## Get the local invariant descriptors

        (kpsA,featuresA) = self.detectKeyPointsAndFeatures(imageA)
        (kpsB,featuresB) = self.detectKeyPointsAndFeatures(imageB)

        match = self.matchKeyPoints(kpsA,featuresA,kpsB,featuresB,ratio,threshold)

        if match is None:
            return None

        ## Unpacking the returned tuples from matchKeyPoints function

        (matches,H,status) = match
        ## Now apply persepective warp on the images
        ## image.shape[1] is added since they represent the number of columns which is the width.
        print(imageA.shape)
        result = cv2.warpPerspective(imageA,H,(imageA.shape[1] + imageB.shape[1],imageA.shape[0]))
        ## Not sure why this is present
        result[0:imageB.shape[0],0:imageB.shape[1]] = imageB

        ## if you need to visualsise
        if showMatches:
            visualise = self.drawMatches(imageA,imageB,kpsA,kpsB,matches,status)
            ## Draw the matches between images and return them with the result
            return(result,visualise)

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

    ## Removing the black area in the image
    def cropImageOnly(self,image):
        ## Convert the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        ## Find threshold
        ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        ## Find contours for finding the original image and removing the black area
        _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        ## Get the first contour
        cnt = contours[0]
        ## Get the rectangle bounding that contour
        x, y, w, h = cv2.boundingRect(cnt)
        ## Crop the image only removing the black area
        cropImage= image[y:y+h,x:x+w]

        return cropImage
        #plt.imshow(img)
        #plt.show()



if __name__ == '__main__':
    imgA = cv2.imread('/home/nikhil/imageStitching/Data/intersection/0.jpg')
    imgB = cv2.imread('/home/nikhil/imageStitching/Data/intersection/1.jpg')
    st = Stitcher()
    #st.visualiseImages(imgA,imgB)
    ##st.drawKeyPoints(imgA)
    gs = gridspec.GridSpec(3, 4)
    gs.update(wspace=0.5)

    stitchImage = st.stitch((imgA,imgB))
    print(stitchImage.shape)
    cropImage = st.cropImageOnly(stitchImage)

    plt.subplot(gs[0,:2]).imshow(imgA)
    plt.subplot(gs[0,2:]).imshow(imgB)
    plt.subplot(gs[1,:]).imshow(stitchImage)
    plt.subplot(gs[2, :]).imshow(cropImage)
    plt.show()
    pass


