from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
import h5py

def singleton(cls):
    _instance = {}

    def inner(*args, **kwgs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwgs)
        return _instance[cls]
    return inner

@singleton
class Datas(object):
    def __init__(self, data_root) -> None:
        self.captions = pkl.load(open(data_root + "/captions.pkl", 'rb'))
        self.tokens2word  = pkl.load(open(data_root+'/token2word.pkl', 'rb'))
        self.train_val_test  = pkl.load(open(data_root+'/train_val_test.pkl', 'rb'))
        self.tokens = pkl.load(open(data_root+'/tokens.pkl', 'rb'))
        self.gyro  =  h5py.File(data_root+'/feats/ivipc_gyro.h5', 'r')
        self.region  =  h5py.File(data_root+'/feats/image_irv2_imagenet_fps10.hdf5', 'r')


class Feat2dDataset(Dataset):
    def __init__(self, data_root, phase, sentence_len=25):
        self.data = Datas(data_root)
        self.phase = phase
        self.splits = self.data.train_val_test[phase]
        self.sentence_len = sentence_len
        
    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx):
        vid = self.splits[idx]
        region  = self.data.region[vid][...]
        gyro = self.data.gyro[vid][...] / np.array([16384, 16384, 16384, 16.4, 16.4, 16.4])
        label = np.zeros((self.sentence_len), dtype=np.int32)
        tmp = np.array(self.data.tokens[vid][0])
        label[:len(tmp)] = tmp
        keyword = np.zeros((500), dtype=np.float32)
        keyword[label] = 1
        keyword[0] = 0
        # print(region.shape, gyro.shape, label.shape)
        return region, gyro, label, keyword

if __name__ == '__main__':
    a = Feat2dDataset(r"E:\datasets\IVIPC",'train')
    b = Feat2dDataset(r"E:\datasets\IVIPC",'test')
    region, gyro, label, keyword = a[0]
    print()
