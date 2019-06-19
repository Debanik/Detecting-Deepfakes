"""
Program to search for video files under it's directory and generate cropped frame images of detected faces
"""
import os
import extract_faces


def search_videos(input_dir=os.path.dirname(os.path.realpath(__file__)), output_dir='~/'):
    
    try:
        os.mkdir(output_dir)
        os.mkdir(output_dir+'/fake')
        os.mkdir(output_dir+'/real')
    except FileExistsError:
        pass
    threshold = int(input("Enter the number of images to be generated per video file: "))
    i = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:

            if file.endswith('.mp4'):
                a = (root + '/' + str(file)).split('/')

                if 'Deepfakes' in a or 'FaceSwap' in a or 'Face2Face' in a:
                    extract_faces.frame_capture(path=(root+'/'+str(file)), output_path=output_dir+'/fake', label='FAKE',
                                                start_counter=i, threshold=threshold)

                else:
                    extract_faces.frame_capture(path=(root + '/' + str(file)), output_path=output_dir+'/real', label='REAL',
                                                start_counter=i, threshold=threshold)

                i += threshold


if __name__ == '__main__':
    input_direc = input("Enter the absolute path of the input directory: ")
    output_direc = input("Enter the absolute path of the output directory: ")
    search_videos(input_direc, output_direc)
