# Kaggle_Understanding-the-Amazon-from-Space

Multi Label satellite image classification

For this project I fine-tuned the InceptionV3 convolutional neural network for the kaggle competition 
https://www.kaggle.com/c/planet-understanding-the-amazon-from-space

Statistics and machine learning techniques:  
Optimize F2 thresholds for each classification    
Test Time data augmentation for running multiple test samples of each image through the model to give the best prediction.   
Training Data Augmentation   
Manual learning rate decay  
k-Fold cross validation-wasn't implemented completely before competition time ran out.  Needed to save off f2 optimization thresholds for each learning model.   


For multi label prediction I used: 
Sigmoid activation at the classification layer with binary cross entropy loss 

Tech:
AWS p2.xlarge ubuntu linux instance (GPU support) 
CUDA/CUDNN Python/Numpy 
Python virtual environment 
Keras 
TensorFlow 
OpenCV 
pandas 
sklearn 

Setting up a computing instance on AWS was a pain. Some of these had various problems I had to google, I'll write down the simple instructions here:

1. Request P2.xlarge instance on aws. You have to write a message to customer support and they'll usually get you approved within a business day. 
2. Create a new Ubuntu p2.xlarge instance on ec2 console. 
3. You'll probably want to extend your partition, especially if you want to hold training data and various ML tech on here. Link for that- 
http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/storage_expand_partition.html 
4. Download CUDA and CUDNN for nvidia GPU support. 
CUDA install command:
sudo apt install nvidia-cuda-toolkit
Link for CUDNN:
https://developer.nvidia.com/rdp/cudnn-download

5. Download python on ec2 instance and figure out how to setup a python virtual environment. Google it. 

6. Download keras. 

7. Download required python libraries such as numpy and pandas. 

8. OpenCV: sudo apt-get install python-opencv. I had some issues getting open CV working, but I googled error and found a good script to run to correct the problem. 

9. Download tensor flow. I'm using pip. Then configure keras to use the tensorflow backend. I tried for a half of a day to get Theano backend working with GPU support and failed miserably. TF pip link:  https://www.tensorflow.org/versions/r0.11/get_started/os_setup#pip_installation 
