import os
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
import tifffile as tiff
import math

class MyDataset(Dataset):
    def __init__(self, Dataset_Root_Path='./dataset/', Subfolder='DS_TNNLS/', TrainingMethod=''):
        self.Dataset_Root_Path = Dataset_Root_Path
        self.Subfolder = Subfolder
        self.TrainingMethod = TrainingMethod
        
        all_content_name = os.listdir(os.path.join(self.Dataset_Root_Path, self.Subfolder))
        if TrainingMethod == 'SL_Datafrom2folder' or TrainingMethod == 'SSL_Datafrom2folder' or TrainingMethod == 'SL_Datafrom2folder_ResNet' or TrainingMethod == 'SSL_Datafrom2folder_ResNet':
            all_content_name = os.listdir(os.path.join(self.Dataset_Root_Path, self.Subfolder, 'rec_amp'))
        if TrainingMethod == 'OffsetLearning':
            all_content_name = os.listdir(os.path.join(self.Dataset_Root_Path, self.Subfolder, 'Input_AePs'))
        if not Subfolder == 'HA_2D_TJ/':
            all_content_name.sort(key= lambda x:int(x[:-4]))
        self.all_content_name_list = all_content_name
    
    def __getitem__(self, index):
        if self.TrainingMethod == 'SSL' or self.TrainingMethod == 'SSL_additionC_NFP' or self.TrainingMethod == 'MultiTask' or self.TrainingMethod == 'SSL_downstream' or self.TrainingMethod == 'SSL_continue' or self.TrainingMethod == 'OffsetDataCollection':
            with open(os.path.join(self.Dataset_Root_Path, self.Subfolder, self.all_content_name_list[index]), 'rb') as Input:
                Ae = tiff.imread(Input) # numpy, (50, 50)
                Ae = Ae.astype(np.float32)
                assert (Ae <= 1.0).all() and (Ae >= 0.0).all(), "Ae is out of range [0, 1]"
                Ae = torch.from_numpy(Ae)
            Ae = Ae.unsqueeze(0)  # torch.Size([1, 50, 50])
            # print(Ae.shape, Ae.max(), Ae.min())
            # exit()
            # assert Ae.shape == torch.Size([1, 50, 50]), "The shapes of Ae is wrong (not [1,50,50])"
            assert (Ae <= 1.0).all() and (Ae >= 0.0).all(), "Ae is out of range [0, 1]"
            return Ae
        elif self.TrainingMethod == 'OffsetLearning':
            with open(os.path.join(self.Dataset_Root_Path, self.Subfolder, 'Input_AePs/', self.all_content_name_list[index]), 'rb') as Input:
                AePs = np.load(Input)
                AePs = AePs.astype(np.float32)
                AePs = torch.from_numpy(AePs)
                assert AePs.shape == torch.Size([2, 50, 50]), "AePs.shape is not (2, 50, 50)"
                assert AePs[1].max() <= 1.0 and AePs[1].min() >= 0.0, "AePs is out of range [0.0, 1.0]"
            with open(os.path.join(self.Dataset_Root_Path, self.Subfolder, 'Lable_PsOffset_direct/', self.all_content_name_list[index]), 'rb') as Label:
                Offset = np.load(Label)
                Offset = Offset.astype(np.float32)
                Offset = torch.from_numpy(Offset)
                assert Offset.shape == torch.Size([1, 50, 50]), "Offset.shape is not (1, 50, 50)"
                assert Offset.max() <= 1.0 and Offset.min() >= -1.0, "AePs is out of range [-1.0, 1.0]"
            return AePs, Offset
        elif self.TrainingMethod == 'SL':
            with open(os.path.join(self.Dataset_Root_Path, self.Subfolder, self.all_content_name_list[index]), 'rb') as Input_Label:
                AePs = np.load(Input_Label)
                AePs = AePs.astype(np.float32)
                AePs = torch.from_numpy(AePs)
                assert AePs.shape == torch.Size([2, 50, 50]), "AePs.shape is not (2, 50, 50)"
                Ae = AePs[0,:,:]
                Ps = AePs[1,:,:]
                assert (Ae <= 1.0).all() and (Ae >= 0.0).all(), "Ae is out of range [0, 1]"
                # assert (Ps <= (2*math.pi)).all() and (Ps >= 0.0).all(), "Ps is out of range [0, 2pi]"
                assert (Ps <= 1.0).all() and (Ps >= 0.0).all(), "Ps is out of range [0, 1]"
                Ae = Ae.unsqueeze(0)  # [1, 50, 50]
                Ps = Ps.unsqueeze(0)  # [1, 50, 50]
                assert Ae.shape == torch.Size([1, 50, 50]) and Ps.shape == torch.Size([1, 50, 50]), "The shapes of Ae and Ps are wrong (not [1,50,50])"
                return Ae, Ps
        elif self.TrainingMethod == 'SL_NIPS':
            with open(os.path.join(self.Dataset_Root_Path, self.Subfolder, 'AC/'), 'rb') as Input:
                AC = tiff.imread(Input)
                AC = AC.astype(np.float32)
                assert (AC <= 1.0).all() and (AC >= 0.0).all(), "AC is out of range [0, 1]"
                AC = torch.from_numpy(AC)
            AC = AC.unsqueeze(0)  # torch.Size([1, 50, 50])
            with open(os.path.join(self.Dataset_Root_Path, self.Subfolder, 'PS/'), 'rb') as Label:
                PS = tiff.imread(Label)
                PS = PS.astype(np.float32)
                assert (PS <= 1.0).all() and (PS >= 0.0).all(), "PS is out of range [0, 1]"
                PS = torch.from_numpy(PS)
            PS = PS.unsqueeze(0)  # torch.Size([1, 50, 50])
            return AC, PS


        # This data loader is used for monitoring how the loss affects the network training with specific training method or among different training methods
        elif self.TrainingMethod == 'SL_Datafrom2folder' or self.TrainingMethod == 'SSL_Datafrom2folder' or self.TrainingMethod == 'SL_Datafrom2folder_ResNet' or self.TrainingMethod == 'SSL_Datafrom2folder_ResNet':
            with open(os.path.join(self.Dataset_Root_Path, self.Subfolder, 'rec_amp/', self.all_content_name_list[index]), 'rb') as Input:
                Ac = tiff.imread(Input)
                Ac = Ac.astype(np.float32)
                assert (Ac <= 1.0).all() and (Ac >= 0.0).all(), "Ac is out of range [0, 1]"
                Ac = torch.from_numpy(Ac)
            Ac = Ac.unsqueeze(0)  # torch.Size([1, 50, 50])
            with open(os.path.join(self.Dataset_Root_Path, self.Subfolder, 'phs/', self.all_content_name_list[index]), 'rb') as Label:
                Ps = tiff.imread(Label)
                Ps = Ps.astype(np.float32)
                # print(Ps.max())
                # exit()
                # Ps = Ps*2*math.pi
                # assert (Ps <= (2*math.pi)).all() and (Ps >= 0.0).all(), "Ps is out of range [0, 2pi]"
                assert (Ps <= 1.0).all() and (Ps >= 0.0).all(), "Ps is out of range [0, 1]"
                Ps = torch.from_numpy(Ps)
            Ps = Ps.unsqueeze(0)  # torch.Size([1, 50, 50])
            assert Ac.shape == torch.Size([1, 50, 50]) and Ps.shape == torch.Size([1, 50, 50]), "The shapes of Ac and Ps are wrong (not [1,50,50])"
            return Ac, Ps
    
    def __len__(self):
        return len(self.all_content_name_list)


def load_data(dataloader_para, train_para):
    
    Dataset_Root_Path = dataloader_para['Dataset_Root_Path']
    Subfolder = dataloader_para['Subfolder']
    TrainingMethod = train_para['training_method']
    batch_size = train_para['batch_size']
    shuffle = dataloader_para['shuffle']
    device = train_para['device']
    nw = dataloader_para['nw']
    data_split_ratio = dataloader_para['data_split_ratio']
    used_dataset_num = dataloader_para['used_dataset_num']
    
    dataset = MyDataset(Dataset_Root_Path=Dataset_Root_Path, Subfolder=Subfolder, TrainingMethod=TrainingMethod)
    Random_Seed = 42
    # Dataset splitting
    # print(len(dataset))
    # exit()
    assert used_dataset_num <= len(dataset), "The dataset is not enough"
    total_data_num = used_dataset_num
    n_train = total_data_num * data_split_ratio[0]
    n_valid = total_data_num * data_split_ratio[1]
    n_test = total_data_num * data_split_ratio[2]

    indices = list(range(total_data_num))
    split_fortrain = int(n_train)
    split_forvalid = int(n_train + n_valid)
    split1 = int(n_train)
    split2 = int(n_train + n_valid)
    if shuffle:
        np.random.seed(Random_Seed)
        np.random.shuffle(indices)
    indices_train, indices_valid, indices_test = indices[:split1], indices[split1:split2], indices[split2:]
    print("Training set : Validation et: Testing set: {} : {} : {}".format(len(indices_train), len(indices_valid), len(indices_test)))

    total_sampler = Data.SubsetRandomSampler(indices)
    train_sampler = Data.SubsetRandomSampler(indices_train)
    valid_sampler = Data.SubsetRandomSampler(indices_valid)
    test_sampler = Data.SubsetRandomSampler(indices_test)
    
    # Calculate running time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    total_loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        sampler=total_sampler,
        # num_workers=nw,
        drop_last=True,
    )
    train_loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        sampler=train_sampler,
        # num_workers=nw,
        drop_last=True,
    )
    valid_loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        sampler=valid_sampler,
        # num_workers=nw,
        drop_last=True,
    )
    test_loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        sampler=test_sampler,
        # num_workers=nw,
        drop_last=True,
    )
    end.record()
    torch.cuda.synchronize()
    print("Using {} ms for dataset loading".format(start.elapsed_time(end)))
    
    return total_loader, train_loader, valid_loader, test_loader

