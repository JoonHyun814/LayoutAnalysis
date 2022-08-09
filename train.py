import dataset
import model

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm


train_data_dir = '../FUNSD/training_data'
test_data_dir = '../FUNSD/testing_data'
east_model_path = 'pths/east_vgg16.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
lr = 0.001
num_epochs = 50

# data
train_dataset = dataset.FUNSD_Dataset(train_data_dir)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size)

test_dataset = dataset.FUNSD_Dataset(test_data_dir)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size)

# model
east_model = model.EAST()
east_model.load_state_dict(torch.load(east_model_path))
east_model.eval()

trans_model = model.LayoutClassification(64,5,8)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(trans_model.parameters(), lr=0.001)


# train
for epoch in range(num_epochs):
    print('*** Epoch {} ***'.format(epoch))

    # Training
    trans_model.train()  
    running_loss, running_acc = 0.0, 0.0
        
    for idx, (img,boxes,gt_mask,gt_class) in tqdm(enumerate(train_dataloader),total=len(train_dataset)//batch_size):
        img.to(device)
        gt_class.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(True):
            outputs = trans_model(img,boxes)
            loss = criterion(outputs[0], gt_class[0])

            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * img.shape[0]
        running_acc += torch.sum(torch.argmax(outputs,axis=2) == gt_class.data)

    running_acc /= (idx+1) * batch_size
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', running_loss, running_acc))

    # Validation
    trans_model.eval()  
    running_acc = 0.0
        
    for idx, (img,boxes,gt_mask,gt_class) in tqdm(enumerate(test_dataloader),total=len(test_dataset)//batch_size):
        img.to(device)
        gt_class.to(device)

        with torch.set_grad_enabled(False):
            outputs = trans_model(img,boxes)

        # statistics
        running_acc += torch.sum(torch.argmax(outputs,axis=2) == gt_class.data)
    running_acc /= (idx+1) * batch_size
    print('{} Acc: {:.4f}\n'.format('valid', running_acc))

    torch.save(trans_model.state_dict(), f'pths/{epoch}_{running_acc}.pth')