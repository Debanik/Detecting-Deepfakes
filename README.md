# Detecting-Deepfakes

In today's world, facial video forgery is widespread and detecting such malpractices is paramount for law enforcement agencies as well as for the protection of any individual's integrity. Advancements in creation of such fake videos has overtaken the possible ways of countermeasuring them.

I was aiming for a way of detecting a type of video forgery known as intra-frame forgery where subjects have their faces replaced or their lip movementss morphed to the particular audio track [1]. I had come across a dataset of videos consisting of facial forgery as well as their real sources [2] on which I carried out my experimentations. Works on this dataset also exist based on Inception Network and Xception Network, but here I have focused on the MobileNet V2 architectures, that can be used in our portable mobile devices, and can be of practical use.

With the rise in technological ways of generating fake videos like DeepFakes and various video editing softwares like Adobe Premier Pro, anyone with a computer at their disposal can misuse such technology to their advantage.

This work is an attempt to classify such videos as Fake or Authentic.

##### © Debanik Banerjee, Master of Technology, National Institute of Technology Karnataka.

[1] Synthesizing Obama: Learning Lip Sync from Audio - Suwajanakorn et. al.
[2] FaceForensics: A Large-scale Video Dataset for Forgery Detection in Human Faces - Rössler et. al.

### Instructions to run the code:
###### Pre-requisite:
Have the dataset and remember the location. You'll need to provide the absolute location address of the source videos as well as the absolute address of the directory where the extracted frames are to be stored

###### 1. Clone the github repo (of course)
_$ git clone https://github.com/Debanik/Detecting-Deepfakes/_
_$ cd Detecting-Deepfakes_

###### 2. Create a virtual environment in Python 3
Install virtualenv
_$ sudo apt install virtualenv_
Create virtual environment
_$ python3 -m virtualenv myenv_

###### 3. Activate the virtualenv
_$ source myenv/bin/activate_

###### 4. Install dependencies
_$pip install -r requirements.txt_

###### 5. Run the code
_$python Basic\ Model\ MobileNet.py_
