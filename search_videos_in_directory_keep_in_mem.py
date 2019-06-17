"""
Program to search for video files under it's directory and generate cropped frame images of detected faces
"""

import os
import extract_faces_keep_in_mem
import cv2


def search_videos(input_dir=__file__):

    dir_path = os.path.dirname(os.path.realpath(input_dir))
    threshold = int(input("Enter the number of images to be generated per video file: "))
    i = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:

            if file.endswith('.mp4'):
                a = (root + '/' + str(file)).split('/')
                if 'Deepfakes' in a or 'FaceSwap' in a or 'Face2Face' in a:
                    extract_faces_keep_in_mem.frame_capture(path=(root+'/'+str(file)), label='FAKE',
                                                            threshold=threshold)
                else:
                    extract_faces_keep_in_mem.frame_capture(path=(root + '/' + str(file)), label='REAL',
                                                            threshold=threshold)
                i += threshold


if __name__ == '__main__':
    input_direc = input("Enter the absolute path of the input directory: ")
    output_direc = input("Enter the absolute path of the output directory: ")
    search_videos(input_direc)
    temp = 0
    for i in extract_faces_keep_in_mem.images:
        cv2.imwrite("%s/%s_%d.jpg" % (output_direc, i[1], temp), i[0])
        temp += 1
