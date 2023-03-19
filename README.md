# ML4SCI-GSoC2023
Application for ML4SCI GSoC 2023 - Vision Transformers for End-to-End Particle Reconstruction for the CMS Experiment
## Task 1
### Architecture
A "VGG-styled" [1] convolutional neural network (CNN) is built: convolution layers of size-3 kernels to expand the channel dimension followed by max-pooling to compress the image size.
### Data cleaning
Few spurious data samples (samples with all zeros) are found and discarded. 
### Data augmentation
Random horizontal and vertical flips are used to expand the small data set. Random rotation was considered but did not improve the testing set performace significantly (< 1% improvement in the AUC).
### Training details
80-20% split for the training and testing set. 10% of the training set is used for validation. Batch size of 32 is used. Adam optimizer [2] with the default setting in pytorch is used with 5 early stopping rounds. Learning rate is reduced by half when the validation AUC stops improving for 2 epochs.
### Results
Similar testing AUC score of 0.812 is achieved with both the pytorch and keras training. Accuracy and loss measurements are also tracked in the notebooks.
## Task 2
### Architecture
A "VGG-styled" convolutional network [1] is built with skip connections [3] for the consecutive convolutional layers with the same output dimensions.
### Data cleaning
It is clear that dimension 1 and 2 correspond to the calorimeter deposits (possibly ECAL and HCAL) but dimension 0 is very noisy in nature and does not seem to exhibit any regular patterns for the two classes (quark or gluon). When dimension 0 is included in the trainings, it leads to instability of the minima reached or the model would collapse to predict one class preferrentially, even when separate CNN is trained for dimension 0. For this reason, dimension 0 is discarded completely. It might be of interest in the future to understand the representation of this dimension (tracker info?).
### Data augmentation
Random horizontal and vertical flips, and rotations are used. 
### Training details
80-20% split for the training and testing set. 10% of the training set is used for validation. Batch size of 64 is used. Adam optimizer [2] with a small learning rate of 1e-4 is used. Learning rate is reduced by 10 times when the validation AUC stops improving for 2 epoch.
### Results
Testing AUC score of 0.787 is achieved.
## Task 3
### Architecture
A basic vision transformer [4] is built. A patch size of 4x4 is used with a latent dimension of 48 for each patch. A hidden dimension of 96 is used for the linear layers. 4 heads are used for the multi-head attention layers and a total of 8 repeating blocks of the transformer structure are used. At the final layer, a global average pooling [5] over the patch dimension is used instead of a CLS token as they are found to provide very similar performance
### Data cleaning.
Same as Task 1.
### Data augmentation
Same as Task 1.
### Training details
80-20% split for the training and testing set. 10% of the training set is used for validation. Batch size of 64 is used. This is larger than Task 1 since the transformer structure is more computationally intensive. Adam optimizer [2] with the default setting in pytorch is used with 10 early stopping rounds. Learning rate is reduced by half when the validation AUC stops improving for 5 epochs.
### Results and discussion
## References:
[1]: Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).  
  
[2]: Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).  
  
[3]: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.  
  
[4]: Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).  
  
[5]: Lin, Min, Qiang Chen, and Shuicheng Yan. "Network in network." arXiv preprint arXiv:1312.4400 (2013).
