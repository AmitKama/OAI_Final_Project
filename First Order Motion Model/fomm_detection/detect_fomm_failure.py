from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from mlxtend.image import extract_face_landmarks
import cv2
from tqdm import tqdm
import numpy as np
import os
import itertools as it

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

def get_distances(landmarks_list):
    distances_list = []
    pbar = tqdm(total=len(landmarks_list))

    for landmarks in landmarks_list:
        distances = []

        # iterate over every combination of landmarks, without repetition
        for a,b in it.combinations(landmarks,2):
            dist = euclidean(a,b)   # calculate euclidean distance
            distances.append(dist)

        # Normalize distances
        distances = [float(i)/sum(distances) for i in distances]

        distances_list.append(distances)
        pbar.update(1)

    pbar.close()
    return distances_list

def display_image_with_landmarks(image, landmarks, k):
    if landmarks is not None:
        for p in landmarks:
            image[p[1]-3:p[1]+3, p[0]-3:p[0]+3, :] = (255, 255, 255)

    cv2.imshow('Frame with landmarks',image)
    cv2.waitKey(70) # ~30 FPS
    cv2.imwrite(f"{k}.png", image)


# Every image outputs landmarks - list of 68 sublists of size 2
# Every landmark outputs a list of distances between every point (2278 elements)
# From a bunch of images - each image has a list of distances
# train on yalefaces dataset
# for every frame in the tested video - resize to 320x243 pixels
# then, predict. If there are any outliers - there's a face warp
k=0
YALEFACE_PATH = "/home/fuzz/OAI_Final_Project/first_order_motion_model/fomm_detection/yalefaces"
print('\n # Collecting landmarks from YaleFaces dataset...\n')

file_list = os.listdir(YALEFACE_PATH)
pbar_train = tqdm(total=len(file_list))
landmarks_list = []
no_face_images_list = []

for filename in os.listdir(YALEFACE_PATH):
    f = os.path.join(YALEFACE_PATH, filename)
    cap = cv2.VideoCapture(f)
    ret, image = cap.read()
    cap.release()
    landmarks = extract_face_landmarks(image)
    pbar_train.update(1)
    if landmarks is None:
        no_face_images_list.append(image)
        continue
    landmarks_list.append(landmarks)
pbar_train.close()

print('\n # Calculating distances between landmarks...\n')
distances_list = np.array(get_distances(landmarks_list))

print('\n # fitting a LOF model to the distances...')
clf = LocalOutlierFactor(novelty=True, n_neighbors=20)
clf.fit(distances_list)


VIDEO_PATH = "/home/fuzz/OAI_Final_Project/first_order_motion_model/fomm/closed_mouth_raise_eyebrows.mp4"

# Init VideoCapture object
reader = cv2.VideoCapture(VIDEO_PATH)
total_frames_in_vid = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

# Define progress bar
pbar = tqdm(total=total_frames_in_vid)

print(f'\n # Collecting landmarks from {VIDEO_PATH}...\n')

frame_count = 0
landmarks_list = []
while reader.isOpened():
    # If finished going over the frames, break
    if frame_count == total_frames_in_vid:
        break
    
    pbar.update(1)

    # Set the video to a specific frame
    vid_location = frame_count
    reader.set(1, vid_location)

    # Read image and get its size
    success, image = reader.read()
    if success == False:
        raise Exception("Something went wrong when reading the video!")

    # Get landmarks
    landmarks = extract_face_landmarks(image)
    
    # Display the current image with landmarks (if there are any)
    display_image_with_landmarks(image,landmarks, k)
    k +=1
    if landmarks is not None:
        landmarks_list.append(landmarks)
    
    frame_count += 1

pbar.close()
cv2.destroyAllWindows()
reader.release()

print('\n # Calculating distances between landmarks...\n')
distances_list = np.array(get_distances(landmarks_list))

preds = []
preds = clf.predict(distances_list) # -1 = outlier

# Anomaly = pre
if preds[preds < 0].size > 0:
    print('\nAt least one frame included a warped face - The video is fake!\n')
else:
    print('\nThere were no frames found with a warped face - The video is real!\n')