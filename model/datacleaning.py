import cv2
from matplotlib import pyplot
import os
import shutil

# --------------------------------------------------------------------------------------------------------------------
# STEP 1: DATA COLLECTION & DATA CLEANING

# OpenCV is an image processing library in python
image = cv2.imread('./test_images/sharapova1.jpg')
print(image.shape)
pyplot.imshow(image)
# pyplot.show()


# The cmap="gray" is a keyword argument passed to pyplot.imshow, which is responsible for mapping a specific colormap
# to the values found in the array that you passed as the first argument.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
pyplot.imshow(gray, cmap='gray')
# pyplot.show()

# The openCV technique 'haarcascades' is used to detect faces, eyes, body parts, etc etc.
# We are gonna use it in our code to get cropped images of our samples

face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

# faces = face_cascade.detectMultiScale(gray)
# print(faces)
# # Thus we get basically a box with face coordinates in our whole image plot
# x, y, w, h = faces[0]
# print(x, y, w, h)
#
# # To draw a rectangle on our image:
# face_in_img = cv2.rectangle(image, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)
# pyplot.imshow(face_in_img)
# pyplot.show()
#
# # destroyAllWindows() simply destroys all the windows we created:
# cv2.destroyAllWindows()
#
# # For getting eyes:
# for (x, y, w, h) in faces:
#     face_in_img = cv2.rectangle(image, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)
#     # To crop only the face(region of interest):
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_face = image[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (eye_x, eye_y, eye_w, eye_h) in eyes:
#         cv2.rectangle(roi_face, (eye_x, eye_y), (eye_x+eye_w, eye_y+eye_h), color=(0, 255, 0), thickness=2)
#
# pyplot.figure()
# pyplot.imshow(roi_face, cmap='gray')
# pyplot.show()


def get_cropped_image(image_path):
    img = cv2.imread(image_path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale)
    for (x, y, w, h) in faces:
        roi_gray = grayscale[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color


# # Test if your function works:
# test_image_path = './test_images/sharapova1.jpg'
# cropped_image = get_cropped_image(test_image_path)
# pyplot.imshow(cropped_image)
# pyplot.show()
# # The following code/image shouldn't show:
# test_image_path = './test_images/sharapova2.JPG'
# cropped_image = get_cropped_image(test_image_path)
# pyplot.imshow(cropped_image)
# pyplot.show()

# Now we got our code ready. All we need to do is specify our path
# and give all of our data stored in our 'datasets' folder to the function

datasets_path = './datasets/'
cropped_data_path = './datasets/cropped_imgs/'

# Import os will do all of our creating and saving into files functionality
img_directory = []
for each_entry in os.scandir(datasets_path):
    if each_entry.is_dir():
        img_directory.append(each_entry.path)

# .isdir() is responsible for checking whether that directory exists.
# .scandir() scans the directory and gives inner directories
# .path gives path of that directory

print(img_directory)

# The Shutil module allows you to do high-level operations on a file, such as copy, create, and remote operations
if os.path.exists(cropped_data_path):
    # rmtree() is used to delete an entire directory tree, path must point to a directory
    shutil.rmtree(cropped_data_path)
os.mkdir(cropped_data_path)

cropped_image_directory = []
celebrity_names = {}

for img_dir in img_directory:
    count = 1
    celebrity_name = img_dir.split('/')[-1]
    print(celebrity_name)

    celebrity_names[celebrity_name] = []

    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image(entry.path)
        if roi_color is not None:
            cropped_folder = cropped_data_path + celebrity_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_directory.append(cropped_folder)
                print("Generating cropped images in folder: ", cropped_folder)

            cropped_img_name = celebrity_name + str(count) + ".png"
            cropped_img_path = cropped_folder + "/" + cropped_img_name

            cv2.imwrite(cropped_img_path, roi_color)
            celebrity_names[celebrity_name].append(cropped_img_path)
            count += 1

# Now we have a clean dataset.
# ------------------------------------------------------------------------------------------------------
# Save the dictionary as we have to use it in image_processing.py
import pickle

# create a dictionary using {}

# save dictionary to person_data.pkl file
with open('cropped_image_directory.pkl', 'wb') as fp:
    pickle.dump(cropped_image_directory, fp)
    print('dictionary saved successfully to file')
# --------------------------------------------------------------------------------------------------------
