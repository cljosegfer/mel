import torch
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from PIL import Image
import wfdb
from tqdm import tqdm
import os
import h5py

class CODE_E_T_Dataset(Dataset):
    def __init__(self, hdf5_file, split_metadata, train_test, transform = None, 
                 tracing_col = 'tracings', output_col = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'], 
                 exam_id_col = 'exam_id', text_col = 'text',
                #  ecg_meta_path, transform=None, **args
                 ):
        # self.ecg_meta_path = ecg_meta_path
        self.hdf5_file = hdf5_file
        # self.mode = args['train_test']
        self.mode = train_test
        # if self.mode == 'train':
        #     self.ecg_data = os.path.join(ecg_meta_path, 'mimic_ecg_train.npy')
        #     self.ecg_data = np.load(self.ecg_data, 'r')
        # else:
        #     self.ecg_data = os.path.join(ecg_meta_path, 'mimic_ecg_val.npy')
        #     self.ecg_data = np.load(self.ecg_data, 'r')

        # self.text_csv = args['text_csv']
        self.metadata = split_metadata

        self.transform = transform

        self.tracing_col = tracing_col
        self.output_col = output_col
        self.exam_id_col = exam_id_col
        self.text_col = text_col

    def __len__(self):
        # return (self.text_csv.shape[0])
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # we have to divide 1000 to get the real value
        # ecg = self.ecg_data[idx]/1000
        ecg = self.hdf5_file[self.tracing_col][self.metadata['index'].loc[idx]].T / 1000
        # ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
        
        # get raw text
        # report = self.text_csv.iloc[idx]['total_report']
        report = self.metadata[self.text_col].loc[idx]

        sample = {'ecg': ecg, 'raw_text': report}

        if self.transform:
            if self.mode == 'train':
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
            else:
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
        return sample

class ECG_TEXT_Dsataset:
    def __init__(self, data_path, seed = 0,
                 val_size = 0.05, tst_size = 0.05, dataset_name = 'CODEmel'):
        self.data_path = data_path
        hdf5_path = os.path.join(self.data_path, 'code15mel.h5')
        metadata_path = os.path.join(self.data_path, 'code15mel.csv')
        self.dataset_name = dataset_name

        print(f'Load {dataset_name} dataset!')
        # self.train_csv = pd.read_csv(os.path.join(self.data_path, 'train.csv'), low_memory=False)
        # self.val_csv = pd.read_csv(os.path.join(self.data_path, 'val.csv'), low_memory=False)
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.metadata = pd.read_csv(metadata_path)

        self.val_size = val_size
        self.tst_size = tst_size

        self.seed = seed
        self.trn_metadata, self.val_metadata, self.tst_metadata = self.split(seed = seed)

        print(f'train size: {self.trn_metadata.shape[0]}')
        print(f'val size: {self.val_metadata.shape[0]}')
        print(f'tst size: {self.tst_metadata.shape[0]}')
        print(f'total size: {self.metadata.shape[0]}')
        
    def get_dataset(self, train_test, T=None):
        if train_test == 'train':
            print('Apply Train-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            print('Apply Val-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        if self.dataset_name == 'CODEmel':
            if train_test == 'train':
                # misc_args = {'train_test': train_test,
                #    'text_csv': self.train_csv,
                #    }
                dataset = CODE_E_T_Dataset(self.hdf5_file, self.trn_metadata, train_test, transform = Transforms)
            elif train_test == 'val':
                # misc_args = {'train_test': train_test,
                #    'text_csv': self.train_csv,
                #    }
                dataset = CODE_E_T_Dataset(self.hdf5_file, self.val_metadata, train_test, transform = Transforms)
            elif train_test == 'test':
                # misc_args = {'train_test': train_test,
                #    'text_csv': self.train_csv,
                #    }
                dataset = CODE_E_T_Dataset(self.hdf5_file, self.tst_metadata, train_test, transform = Transforms)
            else:
                print('pq eu estou no else?', train_test, self.dataset_name)
                # misc_args = {'train_test': train_test,
                #    'text_csv': self.val_csv,
                #    }
                assert False, 'check ds split key'

            # dataset = MIMIC_E_T_Dataset(ecg_meta_path=self.data_path,
            #                            transform=Transforms,
            #                            **misc_args)
            print(f'{train_test} dataset length: ', len(dataset))
        
        return dataset

    def split(self, seed):
        patient_ids = self.metadata['patient_id'].unique()
        np.random.seed(seed)
        np.random.shuffle(patient_ids)

        num_trn = int(len(patient_ids) * (1 - self.tst_size - self.val_size))
        num_val = int(len(patient_ids) * self.val_size)

        trn_ids = set(patient_ids[:num_trn])
        val_ids = set(patient_ids[num_trn : num_trn + num_val])
        tst_ids = set(patient_ids[num_trn + num_val :])

        trn_metadata = self.metadata.loc[self.metadata['patient_id'].isin(trn_ids)].reset_index()
        val_metadata = self.metadata.loc[self.metadata['patient_id'].isin(val_ids)].reset_index()
        tst_metadata = self.metadata.loc[self.metadata['patient_id'].isin(tst_ids)].reset_index()
        self.check_dataleakage(trn_metadata, val_metadata, tst_metadata)

        return trn_metadata, val_metadata, tst_metadata

    def check_dataleakage(self, trn_metadata, val_metadata, tst_metadata):
        trn_ids = set(trn_metadata['exam_id'].unique())
        val_ids = set(val_metadata['exam_id'].unique())
        tst_ids = set(tst_metadata['exam_id'].unique())
        assert (len(trn_ids.intersection(val_ids)) == 0), "Some IDs are present in both train and validation sets."
        assert (len(trn_ids.intersection(tst_ids)) == 0), "Some IDs are present in both train and test sets."
        assert (len(val_ids.intersection(tst_ids)) == 0), "Some IDs are present in both validation and test sets."
