import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
import SimpleITK as sitk

import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    #if train:
        #transforms.append(albumentations.augmentations.geometric.rotate.RandomRotate90())
        #transforms.append(T.RandomRotate(90))  #RandomRotation(90)
        #transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
'''
def rotation(angle, x_min, y_min, x_max, y_max, cx, cy, h, w):
    bbox_tuple = [
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_min),
        (x_max, y_max),
    ] # put x and y coordinates in tuples, we will iterate through the tuples and perform rotation

    rotated_bbox = []
    for i, coord in enumerate(bbox_tuple):
      M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
      cos, sin = abs(M[0, 0]), abs(M[0, 1])
      newW = int((h * sin) + (w * cos))
      newH = int((h * cos) + (w * sin))
      M[0, 2] += (newW / 2) - cx
      M[1, 2] += (newH / 2) - cy
      v = [coord[0], coord[1], 1]
      adjusted_coord = np.dot(M, v)
      rotated_bbox.insert(i, (adjusted_coord[0], adjusted_coord[1]))

    result = [int(x) for t in rotated_bbox for x in t]
    x_max = np.max((result[0],result[2]))
    x_min = np.min((result[0],result[2]))
    y_max = np.max((result[1],result[5]))
    y_min = np.min((result[1],result[5]))

    return x_max, y_max, x_min, y_min
'''
class Dataset(object):
    def __init__(self, root, csv_file, transforms):
        self.root = root
        self.transforms = transforms
        self.data = pd.read_csv(csv_file,sep=',')
        self.data['img_name'] = self.data['img_name'].astype(str)
        self.imgs = list(sorted(os.listdir(root)))
        #self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.imgs = [i for i in self.imgs if i in self.data['img_name'].values]
        # Read only image files in following format
        self.imgs = [i  for i in self.imgs if os.path.splitext(i)[1].lower() in (".mhd", ".mha", ".dcm", ".png", ".jpg", ".jpeg")]   

    def __getitem__(self, idx):
        #img_path = os.path.join(self.root, "images", str(self.imgs[idx]))
        img_path = os.path.join(self.root, str(self.imgs[idx]))
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img)
        #img, size_changes = crop_img_borders_by_edginess(img, width_edgy_threshold=50, dist_edgy_threshold=100)
        nodule_data = self.data[self.data['img_name']==str(self.imgs[idx])]
        num_objs = len(nodule_data)
        boxes = []
        labels = []
        rotate = False
        img = Image.fromarray(img)
        '''
        if random.random() >= 0.5:
            rotate = True
            h, w = img.shape[:2]
            cx, cy = (int(w / 2), int(h / 2))
            img = Image.fromarray(img)
            angle = 90
            img = img.rotate(angle, expand=True)
        else: img = Image.fromarray(img)
        '''
        
        #if nodule_data['label'].any()!=0: # nodule data
        for i in range(num_objs):
            x_min = int(nodule_data.iloc[i]['x'])
            y_min = int(nodule_data.iloc[i]['y'])
            height = int(nodule_data.iloc[i]['height'])
            width = int(nodule_data.iloc[i]['width'])
            y_max = int(y_min+height)
            x_max = int(x_min+width)
            x_scale = 1024 / img.size[0]
            y_scale = 1024 / img.size[1]
            x_min = x_min*x_scale
            y_min = y_min*y_scale
            x_max = x_max*x_scale
            y_max = y_max*y_scale
            boxes.append([x_min, y_min, x_max, y_max])

            if nodule_data.iloc[i]['label']=='draw left hip':
                labels.append(4)
            elif nodule_data.iloc[i]['label']=='draw right hip':
                labels.append(4)
            elif nodule_data.iloc[i]['label']=='L':
                labels.append(1)
            elif nodule_data.iloc[i]['label']=='R':
                labels.append(2)
            elif nodule_data.iloc[i]['label']=='faux profile':
                labels.append(3)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # for non-nodule images
        '''
        else:
            boxes = torch.empty([0,4])
            area = torch.tensor([0])
            labels_ = torch.zeros(0, dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        '''
            
        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = img.resize((1024,1024),resample=0)
        img = Image.fromarray((np.asarray(img)/np.max(img)))  #normalize

        if self.transforms is not None:
            img, target = self.transforms(np.array(img), target)
        
        image_name = str(self.imgs[idx])

        return img, target, image_name

    def __len__(self):
        return len(self.imgs)
    
    
    
    