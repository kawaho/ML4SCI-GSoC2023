# ML4SCI-GSoC2023
Application for ML4SCI GSoC 2023 - Vision Transformers for End-to-End Particle Reconstruction for the CMS Experiment
## Task 1
### Architecture
A "VGG-styled" [1] convolutional neural network (CNN) is built: convolution layers of size-3 kernels to expand the channel dimension followed by max-pooling to compress the image size.
### Data cleaning
Data are normalized according to the means and standard deviations of the training set per channel. Few spurious data samples (samples with all zeros) are found and discarded. 
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
It is clear that dimension 1 and 2 correspond to the calorimeter deposits (possibly ECAL and HCAL) but dimension 0 is very noisy in nature and does not seem to exhibit any regular patterns for the two classes (quark or gluon). When dimension 0 is included in the trainings, it leads to instability of the minima reached or the model would collapse to predict one class preferrentially, even when separate CNN is trained for dimension 0. For this reason, dimension 0 is discarded completely. It might be of interest in the future to understand the representation of this dimension (tracker info?). Otherwise, data are normalized according to the means and standard deviations of the training set per channel. 
### Data augmentation
Random horizontal and vertical flips, and rotations are used. 
### Training details
80-20% split for the training and testing set. 10% of the training set is used for validation. Batch size of 64 is used. Adam optimizer [2] with a small learning rate of 1e-4 is used. Learning rate is reduced by 10 times when the validation AUC stops improving for 2 epoch.
### Results
Testing AUC score of 0.787 is achieved.
## Task 3
### Architecture
A basic vision transformer (ViT) [4] is built. A patch size of 4x4 is used with a latent dimension of 48 for each patch. A hidden dimension of 96 is used for the MLP layers. 4 heads are used for the multi-head attention layers and a total of 8 repeating blocks of the ViT are used. At the final layer, a global average pooling [5] over the patch dimension is used instead of a CLS token as they are found to provide very similar performance
### Data cleaning.
Same as Task 1.
### Data augmentation
Random horizontal and vertical flips, and rotations are used. 
### Training details
80-20% split for the training and testing set. 10% of the training set is used for validation. Batch size of 64 is used. This is made larger than Task 1 as the ViT is more time-consuming to converge. Adam optimizer [2] with the default setting in pytorch is used. Learning rate is reduced by half when the validation AUC stops improving for 5 epochs. Training was terminated when the validation AUC plateaued and a total of 265 epochs were trained. 
### Results and discussion
Testing AUC score of 0.808 is achieved, which is < 1% worse than the performance of the CNN trained in Task 1.  
  
ViT has recently found to outperform traditional CNN in image classifications and other computer vision tasks, possibly due to its ability to access global features in early layers and to strongly propagate the learned representations throughout the whole architecture [6]. However, ViT does not have inductive bias (locality, 2D structure, and translation equivariance) on 2D images as a CNN does. Hence, it needs more data to be trained in comparison. This is observed empirically that adding random rotations augmentation has improved the training performance in this task but not in Task 1, even Task 1 has almost four times more parameters. More layers/ higher latent dimensions/ higher number of heads were attempted but did not result in significant improvements. This could be another indication that the data size avaliable is the limiting factor here. 
  
For electron/photon ID tasks, local energy distribution could be very important since electrons tend to be more spread out along phi because of its charge under the magnet in CMS. In the future, relative position embedding could be considered to better utilize this information.

## References:
[1]: Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).  
  
[2]: Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).  
  
[3]: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.  
  
[4]: Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).  
  
[5]: Lin, Min, Qiang Chen, and Shuicheng Yan. "Network in network." arXiv preprint arXiv:1312.4400 (2013).  
  
[6]: Raghu, Maithra, et al. "Do vision transformers see like convolutional neural networks?." Advances in Neural Information Processing Systems 34 (2021): 12116-12128.
