import joblib
import json
import numpy
import base64
import cv2
from wavelet import cvt_wavelet

# __xyz means a private variable. It is isolated for that particular file.
__name_to_num = {}
__num_to_name = {}
__model = None

def base64_to_cv2_img(base64_img_str):
    encoded_data = base64_img_str.split(',')[1]
    arr = numpy.frombuffer(base64.b64decode(encoded_data), numpy.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# This Python code defines a function called base64_to_cv2_img that takes a Base64 encoded image string as input and
# returns a decoded OpenCV (cv2) image.
# Here's a breakdown of each step in the function:
# 1. encoded_data = base64_img_str.split(',')[1]:
# The input base64_img_str is a Base64 encoded image string. It is typically a long string containing both metadata and
# encoded image data, separated by a comma. The split method is used to split this string at the comma, resulting in a
# list of two parts. The second part, encoded_data, is extracted from the list, which contains the actual Base64 encoded
# image data.
# 2. arr = numpy.frombuffer(base64.b64decode(encoded_data), numpy.uint8):
# This line uses the base64 and numpy libraries to decode the Base64 encoded image data into a numpy array of unsigned
# 8-bit integers (uint8). First, base64.b64decode(encoded_data) decodes the Base64 data into bytes. Then,
# numpy.frombuffer converts these bytes into a numpy array with numpy.uint8 data type, which is suitable for image data
# storage.
# 3. img = cv2.imdecode(arr, cv2.IMREAD_COLOR):
# Here, the cv2.imdecode function from the OpenCV (cv2) library is used to decode the numpy array (previously converted
# from Base64 data) into an actual image. The cv2.IMREAD_COLOR flag indicates that the image should be loaded in color
# (3-channel BGR format).
# 4. return img:
# The function returns the decoded OpenCV image, which can then be used for further image processing or display.
# Overall, this function converts a Base64 encoded image string into an OpenCV image, making it easier to manipulate and
# analyze the image using the functionalities provided by OpenCV.

def get_cropped_image(image_path, base64_image):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = base64_to_cv2_img(base64_image)

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = grayscale[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces


def num_to_name(num):
    return __num_to_name[num]


def classify_image(base64_image, file_path=None):
    cropped_images = get_cropped_image(file_path, base64_image)

    result = []
    for each_img in cropped_images:
        scaled_raw_img = cv2.resize(each_img, (32, 32))
        wvlt_img = cvt_wavelet(each_img, 'db1', 5)
        scaled_wvlt_img = cv2.resize(wvlt_img, (32, 32))
        combined_img = numpy.vstack((scaled_raw_img.reshape(32*32*3, 1), scaled_wvlt_img.reshape(32*32, 1)))

        len_img = 32*32*3 + 32*32
        final_img = combined_img.reshape(1, len_img).astype(float)

        result.append({
            'class': num_to_name(__model.predict(final_img)[0]),
            'class_probability': numpy.around(__model.predict_proba(final_img)*100, 2).tolist()[0],
            'class_dictionary': __name_to_num
        })

    return result

# This Python code defines a function called classify_image, which takes a Base64 encoded image (base64_image)
# and an optional file_path as input. The primary purpose of this function is to classify the content of the image into
# one or more predefined classes using a pre-trained machine learning model. It returns a list of dictionaries
# containing the classification results for the provided image(s).
# Here's a breakdown of the code:
# cropped_images = get_cropped_image(file_path, base64_image):
# This line calls a function called get_cropped_image with the provided file_path and base64_image.
# The purpose of this function is not evident from the code snippet, but it likely processes the image to extract
# relevant regions of interest or crops the image to specific areas. The result is a list of cropped images
# (cropped_images) for further processing.
# result = []:
# This initializes an empty list called result where the classification results will be stored.
# Loop over each cropped image:
# The next few lines run a loop over each image in the list cropped_images, which contains the cropped images:
# a. scaled_raw_img = cv2.resize(each_img, (32, 32)):
# The cv2.resize function from the OpenCV (cv2) library is used to resize each cropped image (each_img) to a new size of
# (32, 32) pixels. This step scales the image to a fixed size to make it suitable for processing by the machine learning
# model.
# b. wvlt_img = cvt_wavelet(each_img, 'db1', 5):
# This line applies a function called cvt_wavelet to each cropped image (each_img) using a wavelet transformation
# (likely using the 'db1' wavelet and five levels of decomposition). This step is likely extracting some relevant
# features from the image for classification purposes.
# c. scaled_wvlt_img = cv2.resize(wvlt_img, (32, 32)):
# Similar to the previous resizing step, this line scales the wavelet-transformed image (wvlt_img) to (32, 32) pixels.
# d. combined_img = numpy.vstack((scaled_raw_img.reshape(32*32*3, 1), scaled_wvlt_img.reshape(32*32, 1))):
# This line stacks the resized raw image (scaled_raw_img) and the resized wavelet-transformed image (scaled_wvlt_img)
# vertically using numpy.vstack. It reshapes both images to column vectors before stacking. The scaled_raw_img is a
# color image, so it's reshaped into a vector of size 32 * 32 * 3 (3 channels - RGB), and scaled_wvlt_img is a grayscale
# image, so it's reshaped into a vector of size 32 * 32 (single channel).
# e. len_img = 32*32*3 + 32*32:
# This variable len_img is assigned the total length of the concatenated image vectors.
# f. final_img = combined_img.reshape(1, len_img).astype(float):
# The combined_img is reshaped into a single row (1D) array of length len_img and then converted to a float data type.
# This array is likely the input data that will be fed into the pre-trained machine learning model for classification.
# g. result.append({ ... }):
# A dictionary containing the classification results for the current image is created and appended to the result list.
# The dictionary includes three items:
# 'class': The predicted class name for the image.
# 'class_probability': A list containing the predicted class probabilities as percentages.
# 'class dictionary': It likely contains a mapping of class names to their corresponding class numbers.
# return result:
# Finally, the function returns the list result, which contains the classification results for all the cropped images
# passed through the loop.


def load_saved_artifacts():
    print('loading saved artifacts...start')
    global __name_to_num
    global __num_to_name
    global __model

    with open('./artifacts/class_dict.json', 'r') as file:
        __name_to_num = json.load(file)
        __num_to_name = {num: name for name, num in __name_to_num.items()}

    if __model is None:
        with open('./artifacts/project2.pkl', 'rb') as file:
            __model = joblib.load(file)
    print('loading saved artifacts...done')


# Just for test, you can check:
def get_b64_for_virat():
    with open('b64.txt') as file:
        return file.read()


if __name__ == '__main__':
    load_saved_artifacts()
    # print(classify_image(get_b64_for_virat(), None))

