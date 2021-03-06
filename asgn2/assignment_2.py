
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
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
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
resnet_input = 32  #size of resnet18 input images


# In[ ]:


# Cersei chose violence, you choose your hyper-parameters wisely using validation data!
batch_size = 2
num_epochs = 5
learning_rate =  0.001
hyp_momentum = 0.9
root_dir = 'Data'


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

      if not os.path.exists(os.path.join(self.root_dir, self.processed_folder, self.training_file)) and not os.path.exists(self.root_dir, os.path.join(self.processed_folder, self.test_file)):
        jamie_bronn_build_dataset()

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

      return img, target

    def get_int_over_union(xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2):
      x_len = min(xmax1,xmax2) - max(xmin1,xmin2)
      y_len = min(ymax1,ymax2) - max(ymin1,ymin2) 

      inter = 0

      if x_len > 0 and y_len > 0:
        inter = x_len * y_len

      union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - inter

      return 1.0 * inter / union

    def is_background(boxes, xmin, ymin, xmax, ymax):

      for box in boxes:
        xmin1 = box[0]
        ymin1 = box[1]
        xmax1 = box[2]
        ymax1 = box[3]

        if get_int_over_union(xmin, ymin, xmax, ymax, xmin1, ymin1, xmax1, ymax1) > 0.5:
          return False

      return True



    def parse_xml(filename):
      tree = et.parse(filename)
      root = tree.getroot()
      object_map = {}
      for obj in root.iter('object'):
        name = obj.find('name').text
        if name not in object_map:
          object_map[name] = []

        temp = {}
        bnd = neighbor.find('bndbox')

        temp['xmin'] = int(bnd.find('xmin').text)
        temp['ymin'] = int(bnd.find('ymin').text)
        temp['xmax'] = int(bnd.find('xmax').text)
        temp['ymax'] = int(bnd.find('ymax').text)

        object_map[name].append(temp)  
        
      return object_map


    def jamie_bronn_build_dataset():
    # Begin
    if not os.path.exists(os.path.join(self.root_dir, self.processed_folder)):
      os.makedirs(os.path.join(self.root_dir, self.processed_folder))    #Create new folder

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    
    for img in os.listdir(os.path.join(self.root_dir, self.training_folder)):
      annotation_file = os.path.join(self.root_dir, self.annotation_training, img.strip('jpg') + 'xml')
      object_map = parse_xml(annotation_file)

      image = imread(os.path.join(self.root_dir, self.training_folder, img))
      image = torch.from_numpy(image)
      image = Image.fromarray(image.numpy(), mode='L')

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

          image2 = image.crop((xmin, ymin, xmax, ymax))

          if self.transform is not None:
            image2 = self.transform(image2)

          train_images.append(image2)
          train_labels.append(name)


    train_labels = np.array(train_labels)
    train_labels = torch.from_numpy(train_labels)

    training_set = (train_images,train_labels)

    for img in os.listdir(os.path.join(self.root_dir, self.test_folder)):
      annotation_file = os.path.join(self.root_dir, self.annotation_test, img.strip('jpg') + 'xml')
      object_map = parse_xml(annotation_file)

      image = imread(os.path.join(self.root_dir, self.test_folder, img))
      width = 

      image = torch.from_numpy(image)
      image = Image.fromarray(image.numpy())

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

          image2 = image.crop((xmin, ymin, xmax, ymax))

          if self.transform is not None:
            image2 = self.transform(image2)

          test_images.append(image2)
          test_labels.append(name)


    test_labels = np.array(test_labels)
    test_labels = torch.from_numpy(test_labels)

    test_set = (test_images,test_labels)

    with open(os.path.join(self.root_dir, self.processed_folder, self.training_file), 'wb') as f:
      torch.save(training_set, f)
    with open(os.path.join(self.root_dir, self.processed_folder, self.test_file), 'wb') as f:
      torch.save(test_set, f)



# ## Train the netwok
# <br/>You can ask Arya to train the network on the created dataset. This will yield a classification network on the 21 classes of the VOC dataset. 

# In[ ]:


composed_transform = transforms.Compose([transforms.Scale((resnet_input,resnet_input)), transforms.ToTensor(),transforms.RandomHorizontalFlip()])
train_dataset = hound_dataset(root_dir=root_dir, train=True, transform=composed_transform) # Supply proper root_dir
test_dataset = hound_dataset(root_dir=root_dir, train=False, transform=composed_transform) # Supply proper root_dir

print('Size of train dataset: %d' % len(train_dataset))
print('Size of test dataset: %d' % len(test_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print ("Checking the training / test set")
def imshow(img):
  npimg = img.numpy()    
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

train_dataiter = iter(train_loader)
train_images, train_labels = train_dataiter.next()
print("Train images")
imshow(torchvision.utils.make_grid(train_images))


test_dataiter = iter(test_loader)
test_images, test_labels = test_dataiter.next()
print("Test images")
imshow(torchvision.utils.make_grid(test_images))

print(test_labels)

# ### Fine-tuning
# Litlefinger has brought you a pre-trained network. Fine-tune the network in the following section:

# In[ ]:

'''
resnet18 = models.resnet18(pretrained=True)

resnet18.fc = nn.Linear(resnet18.fc.in_features, 21)

# Add code for using CUDA here


# In[ ]:


criterion = nn.CrossEntropyLoss()
# Update if any errors occur
optimizer = optim.SGD(resnet18.parameters(), learning_rate, hyp_momentum)


# In[ ]:


def arya_train():
    # Begin


# In[ ]:


# get_ipython().magic(u'time arya_train()')


# # Testing and Accuracy Calculation
# Jorah then asks a question, how is this a detection task?<br/>
# As everybody wonders, Theon Greyjoy suggests a slding window method to test the above trained trained network on the detection task:<br/>
# "We take some windows of varying size and aspect ratios", he mumbled, "and slide it through the test image (considering some stride of pixels) from left to right, and top to bottom, detect the class scores for each of the window, and keep only those which are above a certain threshold value!". "He is right", says Samwell, "I read a similar approach in the paper -Faster RCNN by Ross Girshick in the library, where he uses three diferent scales/sizes and three different aspect ratios, making a total of nine windows per pixel to slide". You need to write the code and use it in testing code to find the predicted boxes and their classes.

# In[ ]:


def theon_sliding_window():
    # Begin


# "Wait", says <b>Jon Snow</b>, "The predicted boxes may be too many and we can't deal with all of them. So, I myself will go and apply non_maximum_supression to reduce the number of boxes". You are free to choose the threshold value for non maximum supression, but choose wisely [0,1].

# In[ ]:


def aegon_targaryen_non_maximum_supression(boxes,threshold = 0.3):
    # 


# Daenerys, the queen, then orders her army to test out the trained model on the test dataset.

# In[ ]:


def daenerys_test(resnet18):
    # Write loops for testing the model on the test set
    # Also print out the accuracy of the model


# In[ ]:


# get_ipython().magic(u'time daenerys_test(resnet18)')


# # Final Showdown
# After covering all the steps and passing the accuracy value to the talking crystal, they all pass through to the land of the living, with a wounded Jon Snow armed with the Dragon-axe. After a fierce battle, Jon Snow manages to go face to face with the Night king. Surrounded by battling men and falling bodies, they engage in a ferocious battle, a battle of spear and axe. After a raging fight, Jon manages to sink the axe into the Night king's heart, but not before he gets wounded by the spear. As dead men fall to bones, Daenerys and others rush to his aid, but it is too late. Everyone is in tears as they look towards the man of honour, Jon Snow, lying in Daenerys's arms when he says his last words: "The night has ended. Winter is finally over!"

# In[ ]:
'''