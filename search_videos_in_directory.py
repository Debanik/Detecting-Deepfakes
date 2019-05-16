"""
Program to search for video files under a directory and generate cropped frame images of detected faces
"""
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

for root, dirs, files in os.walk(dir_path):
    for file in files:

        # change the extension from '.mp3' to
        # the one of your choice.
        if file.endswith('.mp4'):
            a = (root + '\\' + str(file)).split('\\')

            if 'Deepfakes' in a or 'FaceSwap' in a or 'Face2Face' in a:
                print(root + '\\' + str(file) + '\t' + 'FAKE')
            else:
                print(root + '\\' + str(file) + '\t' + 'REAL')
