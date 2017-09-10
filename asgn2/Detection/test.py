from __future__ import division, print_function, unicode_literals
import os 
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from assignment_2_data import hound_dataset
from torch.autograd import Variable
from scipy.ndimage import imread
import xml.etree.ElementTree
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import xml.etree.ElementTree as et
import cPickle as pkl

root_dir = 'Data'
batch_size = 1
model_file_resnet = 'resnet18_model0.001'
window_size = [32, 64, 96, 128]
aspect_ratio = [1, 2, 4, 8]
stride = 8

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

composed_transform = transforms.Compose([transforms.ToTensor()])


# train_dataset = hound_dataset(root_dir=root_dir, train=True, transform=composed_transform) # Supply proper root_dir
# test_dataset = hound_dataset(root_dir=root_dir, train=False, transform=composed_transform) # Supply proper root_dir

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# print('Size of train dataset: %d' % len(train_loader))
# print('Size of test dataset: %d' % len(test_loader))

def parse_xml(filename):
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

def test(model):
        # Write loops for testing the model on the test set
        # You should also print out the accuracy of the model
        correct = 0
        total = 0
        print ("Testing")
        for images, labels in test_loader:
			images = Variable(images)
			outputs = model(images)
			x,predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			print ("x : " + str(x))
			print ("predicted : " + str(predicted) )
			print ("total : " + str(total))
			print ("labels : " + str(labels))
			correct += (predicted.cpu() == labels.cpu()).sum()
			print ("correct : " + str(correct))

			sys.exit()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        return (100 * correct / total)


def test2(model):
	images = os.listdir('Data/VOCdevkit_Test/VOC2007/JPEGImages')
	for image in images:
		if '003612' in image:
			img = Image.open('Data/VOCdevkit_Test/VOC2007/JPEGImages/' +  image)
			img.show()
			print (img.size)
			object_map = parse_xml('Data/VOCdevkit_Test/VOC2007/Annotations/003612.xml')
			# print (object_map)
			for name in object_map:
				if name not in classes:
					continue

				for temp in object_map[name]:				

					xmin = temp['xmin']
					ymin = temp['ymin']
					xmax = temp['xmax']
					ymax = temp['ymax']

				
					img2 = img.crop((xmin,ymin,xmax,ymax))
					img2 = img2.resize((224,224), Image.BILINEAR)
					img2.show()
					img2 = composed_transform(img2)
					print (img2.size())					
					img2 = img2.resize_(1, 3, 224, 224)
					print (img2.size())
					output = model(Variable(img2))
					val,pred = torch.max(output.data, 1)
					print ("val : " + str(val))
					print ("pred : " + str(pred))        			
	           
			w, h = img.size			
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

			sys.exit()
		



resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 21)
resnet18.load_state_dict(torch.load(model_file_resnet))  
test2(resnet18)

