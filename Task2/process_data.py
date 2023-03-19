import math
import pandas as pd
import glob as glob
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa 
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

columns = ['X_jets','y']
batch_size = 4095
counter = 0
ndata = 0
sum_, sumsq_ = 0, 0

def savenp(df, counter, label='Test'):
  dict_ = {}
  sum_, sumsq_ = 0, 0

  X_JETS, Y = np.stack(df['X_jets'].apply(lambda x: np.array(x.tolist()).tolist())), df['y'].to_numpy()
  if label in ['Train', 'Valid']:
    sum_, sumsq_ = np.sum(X_JETS, axis=(0,2,3)), np.sum(X_JETS**2, axis=(0,2,3))
  for idx, (x_jets, y) in enumerate(zip(X_JETS, Y)):
    np.save('./data/%s/%i/file_%i_%i.npy'%(label, int(y), counter, idx), x_jets)
  return sum_, sumsq_

if __name__=='__main__':
  for file_ in glob.glob('./data/*parquet'):
    print(f'Processing {file_}')
    parquet_file = pq.ParquetFile(file_)
    nchunks = math.ceil(parquet_file.num_row_groups/batch_size)
    for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size, columns=columns), total=nchunks):
  
        df = batch.to_pandas()
        df_train_valid, df_test = train_test_split(df, stratify=df['y'], test_size=0.2, random_state=123)
        ndata += len(df_train_valid)

        df_train, df_valid = train_test_split(df_train_valid, stratify=df_train_valid['y'], test_size=0.1, random_state=123)
 
        train_sum_, train_sumsq_ = savenp(df_train, counter, label='Train')   
        valid_sum_, valid_sumsq_ = savenp(df_valid, counter, label='Valid')   
        test_sum_, test_sumsq_ = savenp(df_test, counter)   

        sum_+= train_sum_+valid_sum_
        sumsq_ += train_sumsq_+valid_sumsq_
    
        counter += 1

  mean_ = sum_/(125*125*ndata)
  std_ = np.sqrt( sumsq_/(125*125*ndata) - mean_**2 )

  print('Train+Valid Mean:', mean_)
  print('Train+Valid Std:', std_)
