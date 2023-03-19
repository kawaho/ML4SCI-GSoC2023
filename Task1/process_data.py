import h5py
import numpy as np
from sklearn.model_selection import train_test_split

#read the hdf5 fies
e_set = h5py.File('data/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5', 'r')
p_set = h5py.File('data/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5', 'r')

#convert to np arrays
e_x, p_x = np.asarray(e_set['X']), np.asarray(p_set['X'])

#concat the electon/photon arrays
ep_x = np.concatenate([e_x, p_x])
ep_target = np.concatenate([np.ones(len(e_x)), np.zeros(len(p_x))])

#remove entries with all zeros
nonzeros = np.sum(ep_x, axis=(1,2,3))!=0
ep_x, ep_target = ep_x[nonzeros], ep_target[nonzeros]

#set seed for reproducibility
seed = 123

#split into train (80%) /test (20%) set and save
X_train, X_test, y_train, y_test = train_test_split(ep_x, ep_target, test_size=0.2, stratify=ep_target, random_state=seed)
np.savez('data/egamma.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

#normalize data with training set mean and std to ensure no data leakage
X_train_mean, X_train_std = X_train.mean((0,1,2)), X_train.std((0,1,2))
X_train = (X_train-X_train_mean)/X_train_std
X_test = (X_test-X_train_mean)/X_train_std

#save the normalized set
np.savez('data/egamma_norm.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print(X_train_mean, X_train_std)
