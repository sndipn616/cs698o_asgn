
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
import torch.nn as nn
import torch.utils.data
from scipy.stats import bernoulli
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageOps
from torch.autograd import Variable
from scipy.ndimage import imread
import xml.etree.ElementTree
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import xml.etree.ElementTree as et
import cPickle as pkl

# get_ipython().magic(u'matplotlib inline')
# plt.ion()


# In[ ]:

# with torch.cuda.device(2):
# You can ask Varys to get you more if you desire
resnet_input = 224  #size of resnet18 input images
back_patch_size = 64

# In[ ]:


# Cersei chose violence, you choose your hyper-parameters wisely using validation data!
batch_size = 5
num_epochs = 2
learning_rate =  0.0001
hyp_momentum = 0.9
data_size = 3000
root_dir = 'Data'
back_class = '__background__'
model_file_resnet = 'resnet_model_5classes_weighted'
num_classes = 6
imageset_folder_test = 'VOCdevkit_Test/VOC2007/ImageSets/Main/'


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


classes = ('__background__', 'aeroplane', 'bicycle', 'car', 'cat', 'dog')

# In[ ]:
map_classes = {'__background__' : 0, 'aeroplane' : 1, 'bicycle' : 2, 'car' : 3, 'cat' : 4, 'dog' : 5}

map_classes_inverse = {0 : '__background__',  1 : 'aeroplane', 2 : 'bicycle',  3 : 'car', 4 : 'cat', 5 : 'dog'}


# In[ ]:




class hound_dataset(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
    def __init__(self, root_dir, train, val, transform=None):
      # Begin
      self.train = train
      self.val = val
      self.root_dir = root_dir
      self.transform = transform
      self.training_folder = 'VOCdevkit_Train/VOC2007/JPEGImages'
      self.test_folder = 'VOCdevkit_Test/VOC2007/JPEGImages'
      self.annotation_training = 'VOCdevkit_Train/VOC2007/Annotations'
      self.annotation_test = 'VOCdevkit_Test/VOC2007/Annotations'
      self.imageset_folder_train = 'VOCdevkit_Train/VOC2007/ImageSets/Main/'
      self.imageset_folder_test = 'VOCdevkit_Test/VOC2007/ImageSets/Main/'
      self.training_file = 'training.pt'
      self.validation_file = 'val.pt'
      self.test_file = 'test.pt'
      self.processed_folder = 'Processed'

      if not os.path.exists(os.path.join(self.root_dir, self.processed_folder, self.training_file)) and not os.path.exists( os.path.join(self.root_dir, self.processed_folder, self.test_file)) and not os.path.exists( os.path.join(self.root_dir, self.processed_folder, self.validation_file)):
        self.jamie_bronn_build_dataset()

      if self.train:        
        self.train_data, self.train_labels = torch.load(os.path.join(self.root_dir, self.processed_folder, self.training_file))
      else:
        if self.val:
          self.val_data, self.val_labels = torch.load(os.path.join(self.root_dir, self.processed_folder, self.validation_file))
        else:
          self.test_data, self.test_labels = torch.load(os.path.join(self.root_dir, self.processed_folder, self.test_file))
        
    def __len__(self):
      # Begin
      if self.train:
        return len(self.train_data)
      else:
        if self.val:
          return (len(self.val_data))
        else:
          return len(self.test_data)
        
    def __getitem__(self, idx):
      # Begin
      if self.train:
        img, target = self.train_data[idx], self.train_labels[idx]
      else:
        if self.val:
          img, target = self.val_data[idx], self.val_labels[idx]
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

    def save_dict(self,a,myfile):
      with open(myfile, 'w') as f:
        for key, value in a.items():
          f.write('%s:%s\n' % (key, value))

    def jamie_bronn_build_dataset(self):
      # Begin
      if not os.path.exists(os.path.join(self.root_dir, self.processed_folder)):
        os.makedirs(os.path.join(self.root_dir, self.processed_folder))    #Create new folder

      train_images = []
      train_labels = []
      valid_images = []
      valid_labels = []
      test_images = []
      test_labels = []

      # background_crop = transforms.Compose([transforms.RandomCrop(back_patch_size)])

      index = 0
      train_back = bernoulli.rvs(0.1, size=1000)
      count_classes = {}
      count_classes[back_class] = 0

      print ("Processing Train and Validation Dataset")
      self.train_dict = {'aeroplane' : [], 'bicycle' : [], 'car' : [], 'cat' : [], 'dog' : []}
      self.val_dict = {'aeroplane' : [], 'bicycle' : [], 'car' : [], 'cat' : [], 'dog' : []}
      for files in os.listdir(os.path.join(self.root_dir, self.imageset_folder_train)):
        if '_trainval.txt' in files:
          continue     

        if '_train.txt' in files:
          temp_file = files.split('_train.txt')[0]          
          if temp_file in classes:
            # print (files)

            f = open(os.path.join(self.root_dir, self.imageset_folder_train, files ),'r')
            line = f.readline()
            while (line):
              # if int(line.split(' ')[1]) == 1:
              temp = line.split(' ')
              if len(temp) == 3:
                label = int(temp[2].strip('\n'))
                if label == 1:
                  self.train_dict[temp_file].append(temp[0])

              line = f.readline()


        if '_val.txt' in files:
          temp_file = files.split('_val.txt')[0]          
          if temp_file in classes:
            # print (files)

            f = open(os.path.join(self.root_dir, self.imageset_folder_train, files ),'r')
            line = f.readline()
            while (line):              
              temp = line.split(' ')
              if len(temp) == 3:
                label = int(temp[2].strip('\n'))
                if label == 1:
                  self.val_dict[temp_file].append(temp[0])

              line = f.readline()

      # print (self.train_dict)
      # print (self.val_dict)
      # sys.exit()
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

          if name not in count_classes:
            count_classes[name] = 0

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
            
            # if map_classes[name] == 15:  
              # if train_back[index] == 1:
                # train_images.append(image2)
                # train_labels.append(map_classes[name])
                # count_classes[name] += 1

            # else:
            if img.strip('.jpg') in self.train_dict[name]:
              train_images.append(image2)
              train_labels.append(map_classes[name])
              count_classes[name] += 1
            else:
              valid_images.append(image2)
              valid_labels.append(map_classes[name])

            


        # if train_back[index] == 0:
        #   index = (index + 1) % 1000
        #   continue

        x1,y1,x2,y2,back_image = self.randomCrop(image,back_patch_size)
        # if self.transform is not None:
        if self.is_background(boxes,x1,y1,x2,y2):
          # back_image = self.transform(back_image)
          back_image = Image.fromarray(back_image, mode='RGB')
          if index % 2 == 0: 
            train_images.append(back_image)
            train_labels.append(map_classes[back_class])
            count_classes[back_class] += 1
          else:
            valid_images.append(back_image)
            valid_labels.append(map_classes[back_class])

          
          index = index + 1
        
        # if index == data_size:
        #   break

      train_labels = np.array(train_labels)
      train_labels = torch.from_numpy(train_labels)

      valid_labels = np.array(valid_labels)
      valid_labels = torch.from_numpy(valid_labels)

      training_set = (train_images,train_labels)
      validation_set = (valid_images, valid_labels)

      with open(os.path.join(self.root_dir, self.processed_folder, self.training_file), 'wb') as f:
        torch.save(training_set, f)

      with open(os.path.join(self.root_dir, self.processed_folder, self.validation_file), 'wb') as f:
        torch.save(validation_set, f)

      del training_set
      del train_images
      del train_labels
      del train_back

      del validation_set
      del valid_images
      del valid_labels

      self.save_dict(count_classes,"count_training_classes.txt")

   

      index = 0
      # test_back = bernoulli.rvs(0.05, size=5000)
      count_classes = {}
      count_classes[back_class] = 0
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

          if name not in count_classes:
            count_classes[name] = 0

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
            count_classes[name] += 1

        # if test_back[index] == 0:
        #   index += 1
        #   continue

        x1,y1,x2,y2,back_image = self.randomCrop(image,back_patch_size)
        # if self.transform is not None:
        if self.is_background(boxes,x1,y1,x2,y2):
          # back_image = self.transform(back_image)
          back_image = Image.fromarray(back_image, mode='RGB')
          test_images.append(back_image)
          test_labels.append(map_classes[back_class])   
          count_classes[back_class] += 1     

        index += 1
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
      # del test_back
      self.save_dict(count_classes,"count_test_classes.txt")


# ## Train the netwok
# <br/>You can ask Arya to train the network on the created dataset. This will yield a classification network on the 21 classes of the VOC dataset. 

# In[ ]:

# composed_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])



# composed_transform = transforms.Compose([transforms.Scale((resnet_input,resnet_input)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
# composed_transform = transforms.Compose([transforms.Scale((resnet_input,resnet_input)), transforms.ToTensor()])

# train_dataset = hound_dataset(root_dir=root_dir, train=True, val=False, transform=composed_transform) # Supply proper root_dir
# valid_dataset = hound_dataset(root_dir=root_dir, train=False, val=True, transform=composed_transform) # Supply proper root_dir
# test_dataset = hound_dataset(root_dir=root_dir, train=False, val=False, transform=composed_transform) # Supply proper root_dir

# print('Size of train dataset: %d' % len(train_dataset))
# print('Size of valid dataset: %d' % len(valid_dataset))
# print('Size of test dataset: %d' % len(test_dataset))

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



# print ("Checking the training / test set")
# def imshow(img):
#   npimg = img.numpy()    
#   plt.imshow(np.transpose(npimg, (1, 2, 0)))
#   plt.show()

# train_dataiter = iter(train_loader)
# train_images, train_labels = train_dataiter.next()
# print("Train images")
# print (train_labels.numpy())
# imshow(torchvision.utils.make_grid(train_images))

# valid_dataiter = iter(valid_loader)
# valid_images, valid_labels = valid_dataiter.next()
# print("Validation images")
# print (valid_labels.numpy())
# imshow(torchvision.utils.make_grid(valid_images))

# test_dataiter = iter(test_loader)
# test_images, test_labels = test_dataiter.next()
# print("Test images")
# print(test_labels.numpy())
# imshow(torchvision.utils.make_grid(test_images))

# In[ ]:


# get_ipython().magic(u'time arya_train()')
'''
def return_weights(myfile):
  class_freq = {}
  with open(myfile,'r') as inf:
    r = inf.readline()
    while r:
      class_freq[map_classes[r.split(':')[0]]] = int(r.split(':')[1].strip('\n'))
      r = inf.readline()

  # print class_freq
  max_freq = 0
  for key, value in class_freq.items():
    if value > max_freq:
      max_freq = value

  # print max_freq
  weights = torch.FloatTensor(num_classes)
  
  for key, value in class_freq.items():   
    weights[key] = 1.0 * max_freq / value

  return weights

def save_loss(loss,filename):
  with open(filename, "wb") as fp:
    pkl.dump(loss,fp)

resnet18 = models.resnet18(pretrained=True)

resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
# resnet18.cuda()

w = return_weights('count_training_classes.txt')
# w = w.cuda()

criterion = nn.CrossEntropyLoss(weight=w)
# criterion = nn.CrossEntropyLoss()

optimizer_resnet = torch.optim.SGD(resnet18.parameters(), learning_rate, hyp_momentum) 


def arya_train(sl=0.0001,opt='sgd'):
  # Begin
  print ("Training RESNET18 " + opt)    

  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      images=Variable(images)
      labels=Variable(labels)

      # Forward + Backward + Optimize
      optimizer_resnet.zero_grad()  # zero the gradient buffer          
      outputs = resnet18(images)            

      loss = criterion(outputs, labels)
      loss.backward()
      
      optimizer_resnet.step()

      if (i+1) % 100 == 0:
          print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                 %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
      
              # loss_resnet18.append(loss.data[0])

  # resnet18.save_state_dict(model_file_resnet)
  torch.save(resnet18.state_dict(), model_file_resnet)

def test(model):
    # Write loops for testing the model on the test set
  
  correct = 0
  total = 0

  print ("Testing")
  for images, labels in test_loader:
      images = Variable(images)            
      
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted.cpu() == labels.cpu()).sum()

      # print (correct)
  print('Accuracy of the network after training on test dataset : ' +  str(100.0 * correct / total))
  return (100 * correct / total) 

print ("Batch Size : " + str(batch_size))

# resnet18 = resnet18.train()
# arya_train(learning_rate,'sgd')
# resnet18.load_state_dict(torch.load(model_file_resnet))  
# resnet18 = resnet18.eval()
# acc = test(resnet18)

'''

def relevant_images(val,test,directory):
  test_dict = []
  if test == True:
    split_text = '_test.txt'
  elif val == True:
    split_text = '_val.txt'

  for files in os.listdir(directory):
   
    temp_file = files.split(split_text)[0]          
    if temp_file in classes:
      # print (files)

      f = open(os.path.join(directory, files ),'r')
      line = f.readline()
      while (line):              
        temp = line.split(' ')
        if len(temp) == 3:
          label = int(temp[2].strip('\n'))
          if label == 1:
            test_dict.append(temp[0])

        line = f.readline()

  return test_dict

def theon_sliding_window(model,tune=True):

  if tune == True:
    directory = 'Data/VOCdevkit_Train/VOC2007/'
    dirname = directory + 'ImageSets/Main'
    relevant_image_files = relevant_images(val=True,test=False,directory=dirname)

  else:
    directory = 'Data/VOCdevkit_Test/VOC2007/'
    dirname = directory + 'ImageSets/Main'
    relevant_image_files = relevant_images(val=False,test=True,directory=dirname)

  images = os.listdir(directory + 'JPEGImages')

  window_size = [64, 96, 128]
  aspect_ratio = [1, 1.5, 2]
  stride = 32
  map_cord_to_score = {}
  boxes = []
  for image in images:
    if image.strip('.jpg') not in relevant_image_files:
      continue

    if image != '003931.jpg':
      continue

    print ("name : " + str(image))

    img = Image.open(directory + 'JPEGImages/' +  image)

    img.show()
    img2 = img.copy()

    print ("Sliding Window")
 
    img_width, img_height = img.size
    c = 0
    for wr in window_size:
      for ar in aspect_ratio:
        i = 0
        
        while i < 2:
          if i == 0:
            window_width = wr
            window_height = int(wr*ar)
            if ar == 1:
              i += 2
            else:
              i += 1

          else:
            window_height = wr
            window_width = int(wr*ar)
            if ar == 1:
              i += 2
            else:
              i += 1

          
          ymin = 0
          ymax = ymin + window_height
          flag_y = False
          while ymax < img_height:
            xmin = 0
            xmax = xmin + window_width
            flag_x = False

            while xmax < img_width:
              c += 1

              window_image = img.crop((xmin,ymin,xmax,ymax))
              window_image = window_image.resize((224,224), Image.BILINEAR)
              window_image = new_transform(window_image)
             
              window_image = window_image.resize_(1, 3, 224, 224)
              # print ("Here")
              output = model(Variable(window_image))
              val, predicted = torch.max(output.data, 1)
              val = val.numpy()

              scores = output.data.numpy()
              if predicted[0] != 0:
                prob = getProb(output)
                if prob[0][predicted[0]] > 0.3:

                  temp = []
                  temp.append(xmin)
                  temp.append(ymin)
                  temp.append(xmax)
                  temp.append(ymax)

                  boxes.append(temp)
                  # temp.append(val)
                  # print (val)
                  if val[0] not in map_cord_to_score:
                    map_cord_to_score[val[0]] = temp


                 
                  print ("xmin : " + str(xmin) + " ymin : " + str(ymin) + " xmax : " + str(xmax) + " ymax : " + str(ymax) + " predicted " + str(predicted[0])+  " val : " + str(val[0]) +  " class : " + map_classes_inverse[predicted[0]] + " prob : " + str(prob[0][predicted[0]]) + " count : " + str(c))
              
              if flag_x == True:
                break

              xmin = xmin + stride
              xmax = xmin + window_width
              if xmax >= img_width:
                xmax = img_width - 1
                flag_x = True

            if flag_y == True:
              break

            ymin = ymin + stride
            ymax = ymin + window_height

            if ymax >= img_height:
              ymax = img_height - 1
              flag_y = True


    f = open('Result/' + image.strip('jpg') + 'txt','w')
    draw = ImageDraw.Draw(img2)
    new_boxes = aegon_targaryen_non_maximum_supression(map_cord_to_score, 0.3)    
    # new_boxes = non_max_suppression_fast(boxes, 0.3)
    for new_box in new_boxes:
      x1 = new_box[0]
      y1 = new_box[1]
      x2 = new_box[2]
      y2 = new_box[3]

      img3 = img2.crop((x1,y1,x2,y2))
      img3 = img3.resize((224,224), Image.BILINEAR)
      img3 = new_transform(img3)
      img3 = img3.resize_((1, 3, 224, 224))
      output = model(Variable(img3))
      val, predicted = torch.max(output.data, 1)

      if predicted[0] != 0:
        category = map_classes_inverse[predicted[0]]
        prob = getProb(output)
        if prob[0][predicted[0]] > 0.3:
          f.write(category + "\t")
          f.write(str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + "\n")
          draw.rectangle(((x1, y1), (x2, y2)), outline='green')
          draw.text((x1, y1), category)

    img2.show()
    img2.save('Result/' + image.strip('.jpg') + '_bd.jpg', "JPEG" )
    f.close()

def non_max_suppression_fast(boxes, overlapThresh):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return []

  boxes = np.array(boxes)
  # if the bounding boxes integers, convert them to floats --
  # this is important since we'll be doing a bunch of divisions
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")
 
  # initialize the list of picked indexes 
  pick = []
 
  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]
 
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)
 
  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
    
    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
 
    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]
 
    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > overlapThresh)[0])))
 
  # return only the bounding boxes that were picked using the
  # integer data type
  return boxes[pick].astype("int")


def aegon_targaryen_non_maximum_supression(map_cord_to_score, threshold = 0.3):
  # if there are no boxes, return an empty list
  print ("Non-Maximum Suppression")
  # boxes = np.asarray(boxes)

  # if len(boxes) == 0:
  #   return []
 
  # initialize the list of picked indexes
  pick = []
  # print ("Before Shape : " + str(boxes.shape))
  
  # score = boxes[:,4]
  idxs = []
  # for score, box in sorted(map_cord_to_score.iteritems(), key=lambda (k,v): (v,k)):
  #   idxs.append(box)
  for score, box in sorted(map_cord_to_score.iteritems()):
    # print (score)
    idxs.append(box)

  # grab the coordinates of the bounding boxes
  idxs = np.array(idxs)
  # print (idxs.shape)
  # print (len(idxs))
  x1 = idxs[:,0]
  y1 = idxs[:,1]
  x2 = idxs[:,2]
  y2 = idxs[:,3]

  boxes = idxs
  # reqd_y2 = np.array(reqd_y2)
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  # idxs = np.argsort(score)
  # idxs = np.argsort(reqd_y2)

  # keep looping while some indexes still remain in the indexes
  # list
  # j = 0
  while len(idxs) > 0:
    # grab the last index in the indexes list, add the index
    # value to the list of picked indexes, then initialize
    # the suppression list (i.e. indexes that will be deleted)
    # using the last index
    # print (idxs.shape)
    last = len(idxs) - 1
    i = last
    # print (j)
    # j += 1
    pick.append(i)
    # print (x1[i])
    # print (x1[:last])
    xx1 = np.maximum(x1[i], x1[:last])
    yy1 = np.maximum(y1[i], y1[:last])
    xx2 = np.minimum(x2[i], x2[:last])
    yy2 = np.minimum(y2[i], y2[:last])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
 
    # compute the ratio of overlap
    overlap = (w * h) / area[:last]
 
    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > threshold)[0])), 0)

  return boxes[pick].astype("int")

  

# def daenerys_test(resnet18):
#   return mAP

def sigmoid (x): 
  return 1/(1 + np.exp(-x))

def getProb(output):
  p = output.data.numpy()
  p = sigmoid(p)
  p = p / np.sum(p)

  return p

def iou(xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2):
      x_len = min(xmax1,xmax2) - max(xmin1,xmin2)
      y_len = min(ymax1,ymax2) - max(ymin1,ymin2) 

      inter = 0

      if x_len > 0 and y_len > 0:
        inter = x_len * y_len

      union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - inter

      return 1.0 * inter / union


new_transform = transforms.Compose([transforms.ToTensor()])

resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
resnet18.load_state_dict(torch.load(model_file_resnet, map_location=lambda storage, loc: storage)) 
resnet18 = resnet18.eval()

theon_sliding_window(resnet18,False)
# mAP = daenerys_test(resnet18,False)
# print ("Mean Average Precision : " + str(mAP))