
# coding: utf-8

# # Assignment 2: The Winter is here
# ##### This works best with epic battle music. No spoilers present.
# <br/>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tywin Lannister was right when he said: "The great war is between death and life, ice and fire. If we loose, the night will never end"<br/>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;It has been six months since the white walkers' army marched into the north, led by the night king himself on a dead dragon. It has been a battle like never before: never before have men faced such an enemy in battle, never before have men fought so bravely against a united threat, and never before have they been so gravely defeated.<br />
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; While Cersei is in King's landing, brave men have died fighting the great war. Among others, Tyrion is dead, Arya is dead and Jon Snow is dead, again. In a desperate battle, Daenerys leads all her forces in a final stand-off with the dead just south of Winterfell. <br />
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Her army defeated, she is now on the run on her dragon in an air battle, being chased by two of her own dragons, the Night king and a dead Jon Snow. Suddenly, the Night king's spear hits Danny's dragon, who, raining blood and fire, falls into ice, taking the lost queen, with him. <br />
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Daenerys opens her eyes in a strange place, a place which does not follow the rules of space and time, where the dead souls killed by the dead men are trapped, forever. But who woke her up? There stands near her, Tyrion, with Jorah, Davos, Jon Snow, and everybody else. They all indulge in a heartfelt reunion when someone yells- "But how do we get out?<br />
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Varys sees a talking crystal close by, who asks them of completing a task, which on completion would allow them to go back to the land of the living, with the ultimate tool to defeat the white-walkers and kills the night king, the Dragon-axe. They have summoned you for help, as the task is out of their expertise, to apply a modified CNN to solve the object detection problem on the PASCAL VOC dataset. Varys, the master of whisperers, has used his talents to import the following for you:

# In[ ]:


from __future__ import division, print_function, unicode_literals
import os
import sys
import torch
import random
import torchvision
import numpy as np
import torch.utils.data
from scipy.stats import bernoulli
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.autograd import Variable
from scipy.ndimage import imread
import xml.etree.ElementTree
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import xml.etree.ElementTree as et

# get_ipython().magic(u'matplotlib inline')
# plt.ion()


# In[ ]:


# You can ask Varys to get you more if you desire
resnet_input = 224  #size of resnet18 input images
back_patch_size = 64

# In[ ]:


# Cersei chose violence, you choose your hyper-parameters wisely using validation data!
batch_size = 2
num_epochs = 5
learning_rate =  0.001
hyp_momentum = 0.9
data_size = 3000
root_dir = 'Data'
back_class = '__background__'

# ## Build the data
# The hound who was in charge for getting the data, brought you the following links:
# <br/>Training and validation:
# <br/>http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
# <br/>Testing data:
# <br/>http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# <br/>He also told you that the dataset(datascrolls :P) consists of images from of 20 classes, with detection annotations included. The JPEGImages folder houses the images, and the Annotations folder has the object-wise labels for the objects in one xml file per image. You have to extract the object information, ie. the [xmin, ymin] (the top left x,y co-ordinates) and the [xmax, ymax] (the bottom right x,y co-ordinates) of only the objects belonging to the given 20 classes(aeroplane, bicycle, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, train, TV). For parsing the xml file, you can ask Varys to import xml.etree.ElementTree for you. <br/>
# <br/> You can then ask Bronn and Jamie to organize the data as follows:
# <br/> For every image in the dataset, extract/crop the object patch from the image one by one using their respective co-ordinates:[xmin, ymin, xmax, ymax], resize the image to resnet_input, and store it with its class label information. Do the same for training/validation and test datasets. <br/>
# ##### Important
# You also have to collect data for an extra background class which stands for the class of an object which is not a part of any of the 20 classes. For this, you can crop and resize any random patches from an image. A good idea is to extract patches that have low "intersection over union" with any object present in the image frame from the 20 Pascal VOC classes. The number of background images should be roughly around those of other class objects' images. Hence the total classes turn out to be 21. This is important for applying the sliding window method later.

# In[ ]:


classes = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


# In[ ]:
map_classes = {'__background__' : 0,
           'aeroplane' : 1, 'bicycle' : 2, 'bird' : 3, 'boat' : 4,
           'bottle' : 5, 'bus' : 6, 'car' : 7, 'cat' : 8, 'chair' : 9,
           'cow' : 10, 'diningtable' : 11, 'dog' : 12, 'horse' : 13,
           'motorbike' : 14, 'person' : 15, 'pottedplant' : 16,
           'sheep' : 17, 'sofa' : 18, 'train' : 19, 'tvmonitor' : 20}


# In[ ]:


class hound_dataset(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
    def __init__(self, root_dir, train, transform=None):
      # Begin
      self.train = train
      self.root_dir = root_dir
      self.transform = transform
      self.training_folder = 'VOCdevkit_Train/VOC2007/JPEGImages'
      self.test_folder = 'VOCdevkit_Test/VOC2007/JPEGImages'
      self.annotation_training = 'VOCdevkit_Train/VOC2007/Annotations'
      self.annotation_test = 'VOCdevkit_Test/VOC2007/Annotations'
      self.training_file = 'training.pt'
      self.test_file = 'test.pt'
      self.processed_folder = 'Processed'

      if not os.path.exists(os.path.join(self.root_dir, self.processed_folder, self.training_file)) and not os.path.exists( os.path.join(self.root_dir, self.processed_folder, self.test_file)):
        self.jamie_bronn_build_dataset()

      if self.train:        
        self.train_data, self.train_labels = torch.load(os.path.join(self.root_dir, self.processed_folder, self.training_file))
      else:
        self.test_data, self.test_labels = torch.load(os.path.join(self.root_dir, self.processed_folder, self.test_file))
        
    def __len__(self):
      # Begin
      if self.train:
        return len(self.train_data)
      else:
        return len(self.test_data)
        
    def __getitem__(self, idx):
      # Begin
      if self.train:
        img, target = self.train_data[idx], self.train_labels[idx]
      else:
        img, target = self.test_data[idx], self.test_labels[idx]

      # print ("Before")
      # print (type(img))
      # print (img.size)
      if self.transform is not None:
        img = img.resize((224,224), Image.BILINEAR)
        img = self.transform(img)

      # print ("After")
      # print (type(img))
      # print (img.size())

      return img, target

    def randomCrop(self,img,back_patch_size):
      h, w, c = img.shape
      th = back_patch_size
      tw = back_patch_size

      if w == tw and h == th:
          return img

      x1 = random.randint(0, w - tw)
      y1 = random.randint(0, h - th)
      x2 = x1 + tw
      y2 = y1 + th

      # return x1,y1,x2,y2,img.crop((x1, y1, x2, y2))
      return x1,y1,x2,y2,img[y1:y2,x1:x2,:]


    def get_int_over_union(self,xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2):
      x_len = min(xmax1,xmax2) - max(xmin1,xmin2)
      y_len = min(ymax1,ymax2) - max(ymin1,ymin2) 

      inter = 0

      if x_len > 0 and y_len > 0:
        inter = x_len * y_len

      union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - inter

      return 1.0 * inter / union

    def is_background(self,boxes, xmin, ymin, xmax, ymax):

      for box in boxes:
        xmin1 = box[0]
        ymin1 = box[1]
        xmax1 = box[2]
        ymax1 = box[3]

        if self.get_int_over_union(xmin, ymin, xmax, ymax, xmin1, ymin1, xmax1, ymax1) > 0.5:
          return False

      return True



    def parse_xml(self,filename):
      tree = et.parse(filename)
      root = tree.getroot()
      object_map = {}
      for obj in root.iter('object'):
        name = obj.find('name').text
        if name not in object_map:
          object_map[name] = []

        temp = {}
        bnd = obj.find('bndbox')

        temp['xmin'] = int(bnd.find('xmin').text)
        temp['ymin'] = int(bnd.find('ymin').text)
        temp['xmax'] = int(bnd.find('xmax').text)
        temp['ymax'] = int(bnd.find('ymax').text)

        object_map[name].append(temp)  
        
      return object_map


    def jamie_bronn_build_dataset(self):
      # Begin
      if not os.path.exists(os.path.join(self.root_dir, self.processed_folder)):
        os.makedirs(os.path.join(self.root_dir, self.processed_folder))    #Create new folder

      train_images = []
      train_labels = []
      test_images = []
      test_labels = []

      # background_crop = transforms.Compose([transforms.RandomCrop(back_patch_size)])

      index = 0
      train_back = bernoulli.rvs(0.005, size=5100)
      print ("Processing Train Dataset")
      for img in os.listdir(os.path.join(self.root_dir, self.training_folder)):
        annotation_file = os.path.join(self.root_dir, self.annotation_training, img.strip('jpg') + 'xml')
        object_map = self.parse_xml(annotation_file)

        image = imread(os.path.join(self.root_dir, self.training_folder, img))
        # image = torch.from_numpy(image)
        # image = Image.fromarray(image, mode='RGB')

        boxes = []

        for name in object_map:
          if name not in classes:
            continue

          for temp in object_map[name]:
            temp2 = []

            xmin = temp['xmin']
            ymin = temp['ymin']
            xmax = temp['xmax']
            ymax = temp['ymax']

            temp2.append(xmin)
            temp2.append(ymin)
            temp2.append(xmax)
            temp2.append(ymax)

            boxes.append(temp2)

            # image2 = image.crop((xmin, ymin, xmax, ymax))
            image2 = image[ymin : ymax, xmin : xmax, :]
            image2 = Image.fromarray(image2, mode='RGB')

            # if self.transform is not None:
            #   image2 = self.transform(image2)
            
            if map_classes[name] == 15:  
              if train_back[index] == 1:
                train_images.append(image2)
                train_labels.append(map_classes[name])
            else:
              train_images.append(image2)
              train_labels.append(map_classes[name])


        if train_back[index] == 0:
          index += 1
          continue

        x1,y1,x2,y2,back_image = self.randomCrop(image,back_patch_size)
        # if self.transform is not None:
        if self.is_background(boxes,x1,y1,x2,y2):
          # back_image = self.transform(back_image)
          back_image = Image.fromarray(back_image, mode='RGB')
          train_images.append(back_image)
          train_labels.append(map_classes[back_class])

        # index += 1
        # if index == data_size:
        #   break

      train_labels = np.array(train_labels)
      train_labels = torch.from_numpy(train_labels)

      training_set = (train_images,train_labels)

      with open(os.path.join(self.root_dir, self.processed_folder, self.training_file), 'wb') as f:
        torch.save(training_set, f)

      del training_set
      del train_images
      del train_labels
      del train_back

      index = 0
      test_back = bernoulli.rvs(0.05, size=5000)
      print ("Processing Test Dataset")
      for img in os.listdir(os.path.join(self.root_dir, self.test_folder)):
        annotation_file = os.path.join(self.root_dir, self.annotation_test, img.strip('jpg') + 'xml')
        object_map = self.parse_xml(annotation_file)

        image = imread(os.path.join(self.root_dir, self.test_folder, img))
        height = image.shape[0]
        width = image.shape[1]

        # image = torch.from_numpy(image)
        # image = Image.fromarray(image, mode='RGB')

        for name in object_map:
          if name not in classes:
            continue

          for temp in object_map[name]:
            temp2 = []

            xmin = temp['xmin']
            ymin = temp['ymin']
            xmax = temp['xmax']
            ymax = temp['ymax']

            temp2.append(xmin)
            temp2.append(ymin)
            temp2.append(xmax)
            temp2.append(ymax)

            boxes.append(temp2)

            image2 = image[ymin : ymax, xmin : xmax, :]
            image2 = Image.fromarray(image2, mode='RGB')

            # if self.transform is not None:
            #   image2 = self.transform(image2)              

            test_images.append(image2)
            test_labels.append(map_classes[name])

        if test_back[index] == 0:
          index += 1
          continue

        x1,y1,x2,y2,back_image = self.randomCrop(image,back_patch_size)
        # if self.transform is not None:
        if self.is_background(boxes,x1,y1,x2,y2):
          # back_image = self.transform(back_image)
          back_image = Image.fromarray(back_image, mode='RGB')
          test_images.append(back_image)
          test_labels.append(map_classes[back_class])        

        # index += 1
        # if index == data_size:
        #   break


      test_labels = np.array(test_labels)
      test_labels = torch.from_numpy(test_labels)

      test_set = (test_images,test_labels)
      
      with open(os.path.join(self.root_dir, self.processed_folder, self.test_file), 'wb') as f:
        torch.save(test_set, f)

      del test_set
      del test_images
      del test_labels
      del test_back


# ## Train the netwok
# <br/>You can ask Arya to train the network on the created dataset. This will yield a classification network on the 21 classes of the VOC dataset. 

# In[ ]:

# composed_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

# composed_transform = transforms.Compose([transforms.Scale((resnet_input,resnet_input)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
# # composed_transform = transforms.Compose([transforms.Scale((resnet_input,resnet_input)), transforms.ToTensor()])
# train_dataset = hound_dataset(root_dir=root_dir, train=True, transform=composed_transform) # Supply proper root_dir
# test_dataset = hound_dataset(root_dir=root_dir, train=False, transform=composed_transform) # Supply proper root_dir

# print('Size of train dataset: %d' % len(train_dataset))
# print('Size of test dataset: %d' % len(test_dataset))

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# print ("Checking the training / test set")
# def imshow(img):
#   npimg = img.numpy()    
#   plt.imshow(np.transpose(npimg, (1, 2, 0)))
#   plt.show()

# train_dataiter = iter(train_loader)
# train_images, train_labels = train_dataiter.next()
# print("Train images")
# imshow(torchvision.utils.make_grid(train_images))


# test_dataiter = iter(test_loader)
# test_images, test_labels = test_dataiter.next()
# print("Test images")
# imshow(torchvision.utils.make_grid(test_images))

# print(test_labels)

# In[ ]:


# get_ipython().magic(u'time arya_train()')



