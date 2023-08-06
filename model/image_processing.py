import pywt
from matplotlib import pyplot
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn
import cv2
import joblib
import numpy
import pandas
import os

import joblib
import json
import pickle



# Read dictionary pkl file
with open('cropped_image_directory.pkl', 'rb') as fp:
    cropped_image_directory = pickle.load(fp)
    print(cropped_image_directory)
# ----------------------------------------------------------------------------------------------------
# STEP 2: IMAGE PROCESSING

# For image processing we gonna use feature engineering(extracting features for an image)
# Wavelet transform is just like fourier transform for images


def cvt_wavelet(img, mode='haar', level=1):
    img_array = img
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_array = numpy.float32(img_array)
    # Now we need values in the range 0 and 1 for wavelet transform:
    img_array /= 255
    # db1 <=> haar wavelet
    coefficients = pywt.wavedec2(img_array, mode, level=level)

    coefficients_h = list(coefficients)
    coefficients_h[0] *= 0

    img_array_h = pywt.waverec2(coefficients_h, mode)
    img_array_h *= 255
    img_array_h = numpy.uint8(img_array_h)

    return img_array_h

# The above Python function, w2d, performs a two-dimensional discrete wavelet transform (DWT) on an input image and
# then reconstructs the image after zeroing out the high-frequency wavelet coefficients.
# Import required libraries: The function begins by importing the necessary libraries:
# numpy as np: NumPy is used for numerical operations on arrays.
# pywt: PyWavelets is a Python library used for wavelet transforms.
# cv2: OpenCV is used for computer vision tasks, including loading and manipulating images.
# Define the function: The w2d function takes three arguments:
# img: The input image on which the wavelet transform will be applied.
# mode: The wavelet type used for the transform. Default is 'haar'.
# level: The number of decomposition levels for the wavelet transform. Default is 1.
# Convert the image to grayscale and normalize: The input image img is converted to grayscale and then normalized
# by dividing all pixel values by 255. This step ensures that the pixel values are within the range [0, 1].
# Perform the 2D discrete wavelet transform: The pywt.wavedec2 function is used to perform the 2D DWT on the
# grayscale normalized image. It returns the wavelet coefficients as a tuple.
# Zero out high-frequency coefficients: The function creates a copy of the wavelet coefficients and sets the
# approximation coefficients (coefficients of the lowest frequency subband) to zero. This is effectively removing the
# high-frequency details from the image.
# Reconstruct the image: The modified wavelet coefficients are then used to reconstruct the image using the
# pywt.waverec2 function. This step effectively removes the high-frequency details from the original image.
# Rescale and convert back to uint8: The reconstructed image imArray_H is rescaled by multiplying by 255
# and then converted back to an unsigned 8-bit integer (uint8) data type. This is necessary to display or save the
# image as a standard image format.
# Return the reconstructed image: The function returns the reconstructed image imArray_H.
# In summary, this function applies the 2D discrete wavelet transform to an input image,
# removes high-frequency details, and returns the resulting image with reduced noise and details.
# It can be useful for de-noising or compressing images while preserving important structural information.
# The specific wavelet type and the number of decomposition levels can be adjusted as per the requirements of the
# application.


celebrity_names = {}
for img_dir in cropped_image_directory:
    celebrity_name = img_dir.split('/')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    celebrity_names[celebrity_name] = file_list


class_dict = {}
count = 0
for celebrity_name in celebrity_names.keys():
    class_dict[celebrity_name] = count
    count += 1
print(class_dict)

# Now, we have everything to make training sets

X = []
y = []

for celebrity_name, training_files in celebrity_names.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is None:
            continue
        scaled_raw_img = cv2.resize(img, (32, 32))
        wvlt_img = cvt_wavelet(img, 'db1', 5)
        scaled_wvlt_img = cv2.resize(wvlt_img, (32, 32))
        combined_img = numpy.vstack((scaled_raw_img.reshape(32*32*3, 1), scaled_wvlt_img.reshape(32*32, 1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name])

X = numpy.array(X).reshape(len(X), 4096).astype(float)
# We are basically converting all values to float values(for training model)
# This is because sometimes sklearn gives us lotta warnings with int values
print(X.shape)

# Now images are processed, and our training models are ready for training.
# ----------------------------------------------------------------------------------------------------------------

# STEP 3: TRAINING MODEL

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)

# # Pipeline is basically a mode object(of whichever model you give),
# # but it does the scaling of your input before modelling.
# pipe_model = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10))])
# pipe_model.fit(x_train, y_train)
# print(pipe_model.score(x_test, y_test))
# # You can also get classification report in SVC:
# print(classification_report(y_test, pipe_model.predict(x_test)))

# First, find best model using GridSearchCV:

# pipeline requires naming the steps, manually. make_pipeline names the steps, automatically

model_params = {
    'svm': {
        'model': SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    }
}

# The most commonly used Python solvers for logistic regression are: liblinear: This solver is a library for large-scale
# linear classification and regression. It is optimized for linear models and can handle both L1 and L2 regularization.
# It can also handle multi-class classification problems.

scores = []
best_estimators = {}
for model_name, model in model_params.items():
    pipe_model = make_pipeline(StandardScaler(), model['model'])
    grid_model = GridSearchCV(pipe_model, model['params'], cv=5, return_train_score=False)
    grid_model.fit(x_train, y_train)

    scores.append({
        'model': model_name,
        'best score': grid_model.best_score_,
        'best parameters': grid_model.best_params_,
    })
    best_estimators[model_name] = grid_model.best_estimator_

# Create and dataframe with the grid model so that we can decide which model to use:
model_df = pandas.DataFrame(scores, columns=['model', 'best score', 'best parameters'])
print(model_df)

# NOw, what you did was, you built a dataframe yourself to check which is the best score. But, there is a built-in
# function called best_estimators in GridSearchCV which will give you the score for each model. But it is better to do
# the selection using the make dataframe approach
# To check you can do:
print(best_estimators['svm'].score(x_test, y_test))
print(best_estimators['random_forest'].score(x_test, y_test))
print(best_estimators['logistic_regression'].score(x_test, y_test))


# So we have decided, looking at the scores, that we will use SVM.
# best_model = SVC(kernel='linear', C=1)
# best_model.fit(x_train, y_train)
best_model = best_estimators['svm']

confusion_model = confusion_matrix(y_test, best_model.predict(x_test))
seaborn.heatmap(confusion_model, annot=True)
pyplot.xlabel('Prediction')
pyplot.ylabel('Truth')
pyplot.show()

# -----------------------------------------------------------------------------------------
# STEP 4: EXPORTING THE MODEL:

joblib.dump(best_model, 'project2.pkl')
# Also store the class_dict, it is important
with open('class_dict.json', 'w') as file:
    file.write(json.dumps(class_dict))

# ----------------------------------------------------------------------------------------
