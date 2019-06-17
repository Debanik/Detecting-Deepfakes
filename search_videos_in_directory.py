"""
Program to search for video files under it's directory and generate cropped frame images of detected faces
"""
import os
import extract_faces

dir_path = os.path.dirname(os.path.realpath(__file__))
threshold = int(input("Enter the number of images to be generated per video file: "))
i = 0
images = []
for root, dirs, files in os.walk(dir_path):
    for file in files:

        if file.endswith('.mp4'):
            a = (root + '/' + str(file)).split('/')

            if 'Deepfakes' in a or 'FaceSwap' in a or 'Face2Face' in a:
                extract_faces.frame_capture(path=(root+'/'+str(file)), label='FAKE', start_counter=i,
                                            threshold=threshold)

            else:
                extract_faces.frame_capture(path=(root + '/' + str(file)), label='REAL', start_counter=i,
                                            threshold=threshold)

            i += threshold
