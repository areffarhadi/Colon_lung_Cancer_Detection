# Colon_lung_Cancer_Detection
Transfer Learning in Directed Acyclic Graph and Series Convolutional Neural Network Architectures for Lung and Colon Cancer Detection on Histopathological Images of LC25000 Dataset


**** **Trained models and Further Details will be shared after acceptance of the paper** ****


Two separate files are published for CNN in series (my_train_pretrained_SERIESnet.m) and DAG (my_train_pretrained_DAGnets.m) architectures. The LC25000 dataset for this task must be downloaded from [here](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) and placed on the local hard drive (for example, in drive G). The addresses of the test and training files are placed in trainvaliddata.mat file, which can be used to obtain the results reported in my paper [submitted to journal].

In order to run a sample code using Squeezenet, you can run the Demo_train_pretrained_Squeezenet.m file. It should be noted that for the easier run, all the dataset classes should be placed in a folder and in drive G based on the address file.
