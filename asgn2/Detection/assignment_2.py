
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
from PIL import Image, ImageOps
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
num_epochs = 5
learning_rate =  0.0001
hyp_momentum = 0.9
data_size = 3000
root_dir = 'Data'
back_class = '__background__'
model_file_resnet = 'resnet_model_weighted'

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
      test_images = []
      test_labels = []

      # background_crop = transforms.Compose([transforms.RandomCrop(back_patch_size)])

      index = 0
      train_back = bernoulli.rvs(0.1, size=1000)
      count_classes = {}
      count_classes[back_class] = 0
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
            train_images.append(image2)
            train_labels.append(map_classes[name])
            count_classes[name] += 1


        # if train_back[index] == 0:
        #   index = (index + 1) % 1000
        #   continue

        x1,y1,x2,y2,back_image = self.randomCrop(image,back_patch_size)
        # if self.transform is not None:
        if self.is_background(boxes,x1,y1,x2,y2):
          # back_image = self.transform(back_image)
          back_image = Image.fromarray(back_image, mode='RGB')
          train_images.append(back_image)
          train_labels.append(map_classes[back_class])
          count_classes[back_class] += 1

        index = (index + 1) % 1000
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

composed_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

# composed_transform = transforms.Compose([transforms.Scale((resnet_input,resnet_input)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
# composed_transform = transforms.Compose([transforms.Scale((resnet_input,resnet_input)), transforms.ToTensor()])
train_dataset = hound_dataset(root_dir=root_dir, train=True, transform=composed_transform) # Supply proper root_dir
test_dataset = hound_dataset(root_dir=root_dir, train=False, transform=composed_transform) # Supply proper root_dir

print('Size of train dataset: %d' % len(train_dataset))
print('Size of test dataset: %d' % len(test_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



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

# test_dataiter = iter(test_loader)
# test_images, test_labels = test_dataiter.next()
# print("Test images")
# print(test_labels.numpy())
# imshow(torchvision.utils.make_grid(test_images))

# In[ ]:


# get_ipython().magic(u'time arya_train()')

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
  weights = torch.FloatTensor(21)
  
  for key, value in class_freq.items():   
    weights[key] = 1.0 * max_freq / value

  return weights

def save_loss(loss,filename):
  with open(filename, "wb") as fp:
    pkl.dump(loss,fp)

resnet18 = models.resnet18(pretrained=True)

resnet18.fc = nn.Linear(resnet18.fc.in_features, 21)
# resnet18.cuda()

w = return_weights('count_training_classes.txt')
# w = w.cuda()

criterion = nn.CrossEntropyLoss(weight=w)
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

def theon_sliding_window():
  images = os.listdir('Data/VOCdevkit_Test/VOC2007/JPEGImages')

  for image in images:
    # if '003191' not in image:
    #   continue
    # all_window_images = torch.zeros(1, 3, 224, 224)
    # all_window_images = []
    img = Image.open('Data/VOCdevkit_Test/VOC2007/JPEGImages/' +  image)
    img = img.resize((256,256), Image.BILINEAR)
    img.show()

    img2 = img.copy()    
             
    print ("Sliding Window")
 
    img_width, img_height = img.size
    c = 0
    bs = 0
    draw = ImageDraw.Draw(img2)     
    for wr in window_size:
      for ar in aspect_ratio:
        i = 0
        
        while i < 2:
          if i == 0:
            window_width = wr
            window_height = wr*ar

          else:
            window_height = wr
            window_width = wr*ar

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
              window_image = composed_transform(window_image)
             
              window_image = window_image.resize_(1, 3, 224, 224)
            
              output = model(Variable(window_image))


              if predicted[0] != 0:
                prob = getProb(output)
                if prob[0][predicted[0]] > 0.1:
             
                  draw.rectangle(((xmin, ymin), (xmax, ymax)),outline='red')
                  draw.text((xmin, ymin), map_classes_inverse[predicted[0]])
                  print ("xmin : " + str(xmin) + " ymin : " + str(ymin) + " xmax : " + str(xmax) + " ymax : " + str(ymax) + " predicted " + str(predicted[0])+  " val : " + str(val.numpy()) +  " class : " + map_classes_inverse[predicted[0]] + " prob : " + str(prob[0][predicted[0]]))
              
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
 
      
    img2.show()    


def aegon_targaryen_non_maximum_supression(boxes,threshold = 0.3):
  return 0

def daenerys_test(resnet18):
  return 0

resnet18.load_state_dict(torch.load(model_file_resnet))  
resnet18 = resnet18.eval()
theon_sliding_window()
