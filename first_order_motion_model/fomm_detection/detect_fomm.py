"""
Evaluates a video file with the dlib's face detector to determine whether the video
is real or fake (i.e. DeepFake)

Usage:
python detect_from_video.py
    -i <path to video file>
"""
import argparse
import cv2
import dlib
from tqdm import tqdm
import filetype
import numpy as np

def detect_face_in_frames(video_path):
    """
    Reads a video and evaluates whether the video is real by calculating the amount of frames 
    without a face and the amount with a face. If the ratio between those surpasses the threshold, 
    the video is fake.
    :param video_path: path to video file
    """

    print(f'Starting: {video_path}')

    # Init VideoCapture object
    reader = cv2.VideoCapture(video_path)
    total_frames_in_vid = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Init dlib Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Frame numbers that will be checked
    frames = list(np.linspace(start=0, stop=total_frames_in_vid, endpoint=False))
    pbar = tqdm(total=total_frames_in_vid)

    i=0
    found_face_count = 0
    not_found_face_count = 0
    while reader.isOpened():

        # If finished going over the frames, break
        if i == total_frames_in_vid:
            break
        
        pbar.update(1)

        # Set the video to a specific frame
        vid_location = frames[i]
        reader.set(1, vid_location)

        # Read image and get its size
        success, image = reader.read()
        if success == False:
            raise Exception("Something went wrong when reading the video!")

        # Revert image from BGR (opencv format) to RGB (dlib format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face with upsample factor = 1 (dlib's face detector expects a uint8 ndarray)
        face_list = face_detector(np.uint8(image), 1)

        # Face Found
        if face_list:
            found_face_count += 1
            i = i+1 # increase frame index
        
        # Face not found
        else:
            not_found_face_count += 1
            i = i+1

    pbar.close()
    reader.release()

    face_to_not_face_ratio = not_found_face_count / found_face_count

    # return the ratio between number of frames with face to number of frames without a face
    return face_to_not_face_ratio



THRESHOLD = 0.4

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str, default=None)
    args = p.parse_args()

    video_path = args.video_path

    if video_path is None:
        raise Exception("No video given. Use \"-i\" to specify a video path.")

    if filetype.guess(video_path).mime == "video/mp4":
        ratio = detect_face_in_frames(**vars(args))
        
        if ratio < THRESHOLD:
            print("The not_face_frames / face_frames is {:.3f} which is below the current threshold: {} ==> The video is real!".format(ratio, THRESHOLD))
        else:
            print("The not_face_frames / face_frames is {:.3f} which is above the current threshold: {} ==> The video is fake!".format(ratio, THRESHOLD))
    else:
        raise Exception("Only MP4 format is supported!")