import torch
from torch.utils.data import Dataset,DataLoader
from glob import glob
import os
from tqdm import tqdm
from PIL import Image
import json

import utils
from model import EAST

model_path  = './pths/east_vgg16.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ocr_model = EAST().to(device)
ocr_model.load_state_dict(torch.load(model_path))
ocr_model.eval()

class FUNSD_Dataset(Dataset):
    def __init__(self,data_dir):
        self.image_pathes = sorted(glob(os.path.join(data_dir,'images/*')))
        self.ann_pathes = sorted(glob(os.path.join(data_dir,'annotations/*')))
        self.images = []
        self.boxes_list = []
        self.gt_mask_list = []

        print('loading images...')
        for image_path in tqdm(self.image_pathes):
            img = Image.open(image_path).convert('RGB')
            boxes = utils.detect(img, ocr_model, device)
            self.images.append(img)
            filterd_boxes = []
            for box in boxes:
                x_min,y_min,_,_,x_max,y_max,_,_,score = box
                if x_min <= 0:
                    x_min = torch.tensor(0.)
                if x_max <= 0:
                    x_max = torch.tensor(0.)
                if y_min <= 0:
                    y_min = torch.tensor(0.)
                if y_max <= 0:
                    y_max = torch.tensor(0.)
                if x_max >= img.size[0]:
                    x_max = torch.tensor(img.size[0])
                if x_min >= img.size[0]:
                    x_min = torch.tensor(img.size[0])
                if y_max >= img.size[1]:
                    y_max = torch.tensor(img.size[1])
                if y_min >= img.size[1]:
                    y_min = torch.tensor(img.size[1])
                
                box = torch.tensor([x_min,y_min,x_max,y_min,x_max,y_max,x_min,y_max,score])
                if x_max-x_min > 5 and y_max-y_min > 5:
                    filterd_boxes.append(box.unsqueeze(0))
            self.boxes_list.append(torch.cat(filterd_boxes))

        print('loading annotations...')
        for ann_path in tqdm(self.ann_pathes):
            gt_mask = torch.zeros(img.size[1],img.size[0])
            with open(ann_path,'r') as f:
                y = json.load(f)

            for gt_box_info in y['form']:
                x_min,y_min,x_max,y_max = gt_box_info['box']
                gt_mask[x_min:x_max,y_min:y_max] = utils.labels[gt_box_info['label']]
            self.gt_mask_list.append(gt_mask)
            

    def __len__(self):
        return len(self.image_pathes)


    def __getitem__(self,idx):
        img = self.images[idx]
        boxes = self.boxes_list[idx]
        gt_mask = self.gt_mask_list[idx]
        gt_class = []
        for x_min,y_min,_,_,x_max,y_max,_,_,_ in boxes:
            num_bg = sum(sum(gt_mask[int(x_min):int(x_max),int(y_min):int(y_max)]==0))
            num_other = sum(sum(gt_mask[int(x_min):int(x_max),int(y_min):int(y_max)]==1))
            num_header = sum(sum(gt_mask[int(x_min):int(x_max),int(y_min):int(y_max)]==2))
            num_question = sum(sum(gt_mask[int(x_min):int(x_max),int(y_min):int(y_max)]==3))
            num_answer = sum(sum(gt_mask[int(x_min):int(x_max),int(y_min):int(y_max)]==4))
            score_list = [num_bg,num_other,num_header,num_question,num_answer]
            class_num = score_list.index(max(score_list))
            gt_class.append(class_num)

        return utils.load_pil(img), self.boxes_list[idx], self.gt_mask_list[idx], torch.tensor(gt_class)


if __name__ == '__main__':
    test_data_dir = '../FUNSD/testing_data'
    test_dataset = FUNSD_Dataset(test_data_dir)
    test_idx = 0

    print(len(test_dataset))
    img,boxes,gt_mask,gt_class = test_dataset[test_idx]
    print(test_dataset[test_idx])
    print(img.shape)
    print(boxes.shape)
    print(gt_class.shape)
    print(gt_mask.shape)

    test_dataloader = DataLoader(test_dataset,batch_size=1)
    img,boxes,gt_mask,gt_class = next(iter(test_dataloader))
    print(img.shape)
    print(boxes.shape)
    print(gt_class.shape)
    print(gt_mask.shape)
