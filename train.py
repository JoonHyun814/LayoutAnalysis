import dataset
import model
from torch.utils.data import DataLoader


train_data_dir = '../FUNSD/training_data'
test_data_dir = '../FUNSD/testing_data'
batch_size = 4
Epoch = 10

train_dataset = dataset.FUNSD_Dataset(train_data_dir)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size)

train_dataset = dataset.FUNSD_Dataset(train_data_dir)
train_dataloader = DataLoader(train_dataset,batch_size=1)