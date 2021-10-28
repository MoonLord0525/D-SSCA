import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from utils.Embedding import one_hot

"""
    文件的命名规范如下：
        对于Sequence文件夹：Train_seq.csv或者Test_seq.csv
        对于Shape文件夹：Train_shape-name.csv或者Test_shape-name.csv
    
    一级文件夹的命名与ENCODE中的保持一致，例如：
        wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk
    
    各个一级文件夹中，有两个二级文件夹，即Sequence文件夹和Shape文件夹
    
"""


class SampleReader:
    """
        SampleReader一次可读取一个文件夹下的一些文件，具体策略如下：
            get_seq()函数可以读取Sequence文件夹中有关的文件
            get_shape()函数可以读取Shape文件夹中有关的文件

        注：对于Train和Test，不能同时读取
    """

    def __init__(self, file_name):
        """
            file_path:
                wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk
        """
        self.seq_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '\\' + file_name + '\\Sequence\\'
        self.shape_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '\\' + file_name + '\\Shape\\'

    def get_seq(self, Test=False):

        if Test is False:
            row_seq = pd.read_csv(self.seq_path + 'Train_seq.csv', sep=' ', header=None)
        else:
            row_seq = pd.read_csv(self.seq_path + 'Test_seq.csv', sep=' ', header=None)

        seq_num = row_seq.shape[0]
        seq_len = len(row_seq.loc[0, 1])

        completed_seqs = np.empty(shape=(seq_num, seq_len, 4))
        completed_labels = np.empty(shape=(seq_num, 1))
        for i in range(seq_num):
            completed_seqs[i] = one_hot(row_seq.loc[i, 1])
            completed_labels[i] = row_seq.loc[i, 2]
        completed_seqs = np.transpose(completed_seqs, [0, 2, 1])

        return completed_seqs, completed_labels

    def get_shape(self, shapes, Test=False):

        shape_series = []

        if Test is False:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Train' + '_' + shape + '.csv'))
        else:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Test' + '_' + shape + '.csv'))

        """
            seq_num = shape_series[0].shape[0]
            seq_len = shape_series[0].shape[1]
        """
        completed_shape = np.empty(shape=(shape_series[0].shape[0], len(shapes), shape_series[0].shape[1]))

        for i in range(len(shapes)):
            shape_samples = shape_series[i]
            for m in range(shape_samples.shape[1]):
                completed_shape[m][i] = shape_samples.loc[m]
        completed_shape = np.nan_to_num(completed_shape)

        return completed_shape


class SSDataset_690(Dataset):

    def __init__(self, file_name, Test=False):
        shapes = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']

        sample_reader = SampleReader(file_name=file_name)

        self.completed_seqs, self.completed_labels = sample_reader.get_seq(Test=Test)
        self.completed_shape = sample_reader.get_shape(shapes=shapes, Test=Test)

    def __getitem__(self, item):
        return self.completed_seqs[item], self.completed_shape[item], self.completed_labels[item]

    def __len__(self):
        return self.completed_seqs.shape[0]