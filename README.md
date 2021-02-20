## This repository is an example of work I have completed, due to data protection the data used to train and test many models cannot be uploaded.

#### [Computer Assisted Surgery and Therapy](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Computer%20Assisted%20Surgery%20and%20Therapy)
Computer assisted surgery focused around the design and deployment of computer guided tools in clinical environments. 
The code shown serves to simulate and analyse one such system and how accuracy is affected by several different scenarios


#### [Dissertation](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Dissertation)
My dissertation involved the development of a quality control tool for analysing organ delineation and contouring before treatment
this was done by developing a Convolutional Neural Network (CNN) which could accept 3D patches. Each patch was subsequently classified 
as to whether or not it contained one of the three most commonly occuring contour artefacts as outlined by clinitians. The full report can be found [here](https://github.com/Jack-Weeks/Coding_Portfolio/blob/main/Dissertation/RT_Seg_Final_Report.pdf)
 
- [ExtractingPatches.py](https://github.com/Jack-Weeks/Coding_Portfolio/blob/main/Dissertation/Extracting%20Patches.py) -
Is the script used to convert the large Nifti slice structures into smaller numpy array patches.

- [Dataloader.py](https://github.com/Jack-Weeks/Coding_Portfolio/blob/main/Dissertation/DataLoader.py) - Contains the 
method used to feed the numpy array patches into the network whilst maintaining the 3D nature of the data.


- [Models.py](https://github.com/Jack-Weeks/Coding_Portfolio/blob/main/Dissertation/Models.py) - Uses as adapted form of the 
pre-defined Vgg models in Pytorch to accept 3D inputs and other slight modifications for the specified task.

- [Training.py](https://github.com/Jack-Weeks/Coding_Portfolio/blob/main/Dissertation/Training.py) - Contains the main script used in
 training the model.
 
#### [Information Processing](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Information%20Processing)

Information Processing in Medical Imaging (IPMI) Focused on the data pipeline from images recorded from medical devices 
into practical use. Coding tasks were centred around the registration of two or more images to one another and Image segmentation algorithms

#### [Machine Learning](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Machine%20Learning)
- [Face and Gender Classification](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Machine%20Learning/Face%2BGender%20Classification) - 
Use transfer learning to develop classifiers to classify celebrity gender and cartoon face shapes.

- [Segmentation Comparison](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Machine%20Learning/Segmentation%20Comparison) - 
Developed a U-Net which could take 2D slices or the 3D volume as an input and analyse which method is better in terms of performance and speed

#### [Research Software Engineering in Python](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Software%20Engineering)
Learning best software development practices including use of Git, TravisCI, Pep8, Code Testing etc.
- [Creating a Python Package](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Software%20Engineering/Creating%20Python%20Package) - Tasked to develop a package which could plan routes for a given city
- [Data Clustering Algorithm](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Software%20Engineering/Data%20Clustering%20Algorithm) -
Analysing and optimising a basic clustering algorithm.

#### [Undergraduate](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Undergraduate)
Various tasks completed as part of my undergraduate course.

- [Angry Birds](https://github.com/Jack-Weeks/Coding_Portfolio/blob/main/Undergraduate/Mechanics_Angry_birds.ipynb) - Tasked with 
making a simplified version of angry birds with followed the laws of motion - 1st year project
- [Higgs Boson Notebooks](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Undergraduate/Higgs-Boson%20Notebooks) -
Examples of Support vector machine and Multi Layer Perceptron algorithms from Sckit-learn to analyse and classify Higgs Boson Signals - 3rd Year group project


#### [Tennis Social Media Portfolio](https://github.com/Jack-Weeks/Coding_Portfolio/tree/main/Tennis%20Social%20Media%20Portfolio)
- Couple of examples of creative approaches to social media marketing 

