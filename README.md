# P5 - Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/CarvsNonCar.png
[image2]: ./output_images/CarvsNonCar_RGB_HSV_HLS_LUV.png
[image3]: ./output_images/CarvsNonCar_Hog.png
[image4]: ./output_images/Car_Raw_Normalized.png
[image5]: ./output_images/PredictvsDecisionFunction.png
[image6]: ./output_images/TestWindows_Drawn.png
[image7]: ./output_images/window_heatmap_thresholdheatmap_finalimg.png
[video1]: ./project_video_final.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. 

For details of the steps taken in this project and the codes, please refer to P5.ipynb.

The project was started by reading in all the 'car' and 'non car' images to consider the size of dataset. This is done to ensure a balance data set. Below is a read out of the data.

* Number of car images: 8792 with shape (64, 64, 3)
* Number of non car images: 9666 with shape (64, 64, 3)

Sampling of 50 cars and 50 non cars were then displayed to verify the data set read. Below is an example of the 'car' and 'non car' classes:

![alt text][image1]

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

To explore the different color spaces and to determine which color space provides the greatest differentiation between car and non car images, scatter plot of car and non car images are plotted for RGB, HSV, HLS and LUV for visualization of car and non car images in different color spaces. In this case, HSV is selected.

![alt text][image2]

Hog feature extraction can be found in Step 4 of 'P5.ipynb'. 

Different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) were explored that were appropriate for a 64x64 image, since that is the size of the image used for training. Random images from each of the two classes were grabbed and displayed to visualize the hog features output for car and non car images. After some trial and error with the parameters, the final Hog parameters selected were `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` as it provided noticeable difference in appearance between car and non car features.

![alt text][image3]

####2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Step 5 - 7 of 'P5.ipynb' documents the data preparation process and training of the classifier.

The final features used for classification were `spatial binning`, `color histogram` and `hog features`. These features were concatenated and normalized to prevent any of the 3 classification with larger magnitude from dominating during the classification. Results image, raw features and normalized features can be seen in the picture below.

![alt text][image4]

Car and non car images were combined and randomized prior to input for classification. 10% of the combined set were also kept aside for the testing of the training. Below shows the final number of test cases after separation.

* Size of Training Set: 16612
* Size of Test Set: 1846

Linear SVM was used in the classification where below shows the results of the classification. 

* 99.56 Seconds to train SVC...
* Classifier class order: [0 1]
* 8.9 Seconds to make prediction using simple prediction on test sample...
* 8.88 Seconds to make prediction using decision function on test sample...
* Test Accuracy of SVC for color space using prediction HSV  =  0.9865
* Test Accuracy of SVC for color space using decision function HSV  =  0.9816

Decision function was also used on the test cases where only a decision threshold of > 0.5 will be classified as 'car'. Even though the test accuracy was lower using decision function, decision function was effective in reducing the number of false positives where the results can be seen in the next picture.

![alt text][image5]

---

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

Step 8 of 'P5.ipynb' documents the code for sliding window search.

Sliding window search was designed with 4 different window size and to slide across different range of the image. Below shows a summary of the design.

* 64x64 : x_start_stop=[300, None], y_start_stop=[400, 464]
* 128x128 : x_start_stop=[None, None], y_start_stop=[375, 427]
* 256x256 : x_start_stop=[None, None], y_start_stop=[315, 418]
* 420x420 : x_start_stop=[None, None], y_start_stop=[250, 350]

Overlap was chosen at 80% in order to reliably capture all the cars without overly slowing down the computation.

Different colors of the 4 windows were drawn on the test images to visualize and affirm the correct implementation. Results can be seen in the picture below.

![alt text][image6]

####2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to try to minimize false positives and reliably detect cars?

As mentioned in earlier question, decision function > 0.5 was used to minimize the number of false positives and yet being able to reliably detect cars. Heatmap thresholding of 1 where 1 or less overlaps being rejected will also reduce the number of false positives. Heatmap thresholding will be increased to 3 during video processing due to the averaging across frames of video.

Below is picture of the detection with heatmaps shown for troubleshooting images.

![alt text][image7]

---

### Video Implementation

####1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to the output video](./project_video_final.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Step 16 of 'P5.ipynb' docuements the creation of `CarTracking` class used in the tracking of cars across frames of video. 

Tracking of car is done across the last 5 frames of the video in order to provide a more stable and consistent detection with rejection of false positives. 5 frames were chosen as the video was recorded in 25fps which makes 5 frames at 0.2s with little change in car position. The raw heat maps from the 5 frames are summed up and threshold at 3 where areas with 3 or less overlapping boxes will be rejected. 

Here's an example result showing the heatmap and bounding boxes overlaid on a frame of video (same image as figure above):

![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

During the implementation of the project, there is a fine balance between rejecting false positives and generating significant number of heatmaps where a car can be accurately detected and box of the right size drawn over the vehicle. Though the current algorithm is able to perform reasonably, improvements can still be made by using the concept of 'Blob Detection' and 'Watershed Segmentation'.

Blob detection in Sci-kit Image (Determinant of a Hessian [`skimage.feature.blob_doh()`](http://scikit-image.org/docs/dev/auto_examples/plot_blob.html)) can be used to identify individual blobs in the heatmap. To segment blobs which are close together which represent 2 different cars, [`skimage.morphology.watershed()`](http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html) can be used to identify the 2 blobs. Bounding boxes can then be constructed to cover the area of each blob detected. This will likely give a more accurate separation of cars that are positioned close together and more accurate size of bounding boxes.
