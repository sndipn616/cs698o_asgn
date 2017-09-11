from __future__ import division, print_function, unicode_literals
import os 
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
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
batch_size = 10
model_file_resnet = '3_resnet18_model0.001sgd'
window_size = [64, 96, 128]
aspect_ratio = [1, 2, 4]
stride = 32

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

map_classes_inverse = {0 : '__background__',
           1 : 'aeroplane', 2 : 'bicycle', 3 : 'bird', 4 : 'boat',
           5 : 'bottle', 6 : 'bus', 7 : 'car', 8 : 'cat', 9 : 'chair',
           10 : 'cow', 11 : 'diningtable', 12 : 'dog', 13 : 'horse',
           14 : 'motorbike', 15 : 'person', 16 : 'pottedplant',
           17 : 'sheep', 18 : 'sofa', 19 : 'train', 20 : 'tvmonitor'}

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
    count = {}
    count2 = {}
    print ("Testing")
    for images, labels in test_loader:
		# img = images.numpy()
		# img = img.reshape(224, 224, 3)
		# print (img.shape)
		# img = Image.fromarray(img)
		# img.show()
		# print (type(images))
		# img = images.resize_(224,224, 3)
		# img = Image.fromarray(img.numpy())
		# img.show()
		# print (images)
		images = Variable(images, requires_grad=True)
		outputs = model(images)
		# print (outputs.size())
		x,predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		# print ("x : " + str(x))
		# print ("predicted : " + str(predicted.numpy()) )
		# print ("total : " + str(total))
		# print ("labels : " + str(labels[0]))
		predicted2 = predicted.numpy()
		labels2 = labels.numpy()
		print ("Predicted : " + str(predicted2) + " Actual : " + str(labels2) + " Correct : " + str(predicted2 == labels2))
		for i in predicted2:
			if i not in count:
				count[i] = 0

		for i in labels2:
			if i not in count2:
				count2[i] = 0	

		count[predicted[0]] += 1
		count2[labels[0]] += 1

		correct += (predicted.cpu() == labels.cpu()).sum()
		# break
		# print ("correct : " + str(correct))

		# sys.exit()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return (100 * correct / total), count, count2


def test2(model):
	images = os.listdir('Data/VOCdevkit_Test/VOC2007/JPEGImages')

	for image in images:
		if '004115' not in image:
			continue
		all_window_images = torch.zeros(1, 3, 224, 224)
		# all_window_images = []
		img = Image.open('Data/VOCdevkit_Test/VOC2007/JPEGImages/' +  image)
		# img = imread('Data/VOCdevkit_Test/VOC2007/JPEGImages/' +  image)
		# img.show()
		# print (img.size)
		object_map = parse_xml('Data/VOCdevkit_Test/VOC2007/Annotations/' + image.strip('.jpg') + '.xml')
		# print (object_map)
		for name in object_map:
			# if name not in classes:
			# 	continue

			# for temp in object_map[name]:				

			# 	xmin = temp['xmin']
			# 	ymin = temp['ymin']
			# 	xmax = temp['xmax']
			# 	ymax = temp['ymax']

			
			# 	img2 = img.crop((xmin,ymin,xmax,ymax))
			# 	img2 = img2.resize((224,224), Image.BILINEAR)
			# 	# img2.show()
			# 	img2 = composed_transform(img2)
			# 	# print (img2.size())					
			# 	img2 = img2.resize_(1, 3, 224, 224)
			# 	# print (img2.size())
			# 	output = model(Variable(img2))
			# 	val,pred = torch.max(output.data, 1)
			# 	# print ("val : " + str(val))
			# 	print ("pred : " + str(map_classes_inverse[pred[0]]))        			
	           
			print ("Sliding Window")
			# print (img.shape)
			# img_height, img_width, img_channel = img.shape
			# sys.exit()
			img_width, img_height = img.size
			c = 0			
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
					 	while ymax < img_height:
							xmin = 0
							xmax = xmin + window_width
							

							while xmax < img_width:
								c += 1
								window_image = img.crop((xmin,ymin,xmax,ymax))
								window_image = window_image.resize((224,224), Image.BILINEAR)
								window_image = composed_transform(window_image)
								# print (window_image.size())
								window_image = window_image.resize_(1, 3, 224, 224)
								# window_image = img[ymin:ymax,xmin:xmax,:]
								# all_window_images.append(window_image)
								all_window_images =  torch.cat([all_window_images, window_image])
								# window_image.show()
								# output = model(Variable(window_image))
								# val, predicted = torch.max(output.data, 1)
								# print (predicted[0])
								# if predicted[0] in map_classes_inverse:
								# print ("xmin : " + str(xmin) + " ymin : " + str(ymin) + " xmax : " + str(xmax) + " ymax : " + str(ymax) + " predicted " + str(predicted[0]) +  " class : " + map_classes_inverse[predicted[0]])
								# else:
								# 	print ("xmin : " + str(xmin) + " ymin : " + str(ymin) + " xmax : " + str(xmax) + " ymax : " + str(ymax) + " predicted " + str(predicted[0]) + " class : Unknown ")
								print ("xmin : " + str(xmin) + " ymin : " + str(ymin) + " xmax : " + str(xmax) + " ymax : " + str(ymax) + " count : " + str(c))
								xmin = xmin + stride
								xmax = xmin + window_width

							# print ("Vertical")
							ymin = ymin + stride
							ymax = ymin + window_height
				# sys.exit()

				        	# ymax = ymin + window_height

			output = model(Variable(all_window_images))
			val, predicted = torch.max(output.data, 1)
			print (predicted)
			# all_window_images = torch.FloatTensor(all_window_images)
			# all_window_images = np.asarray(all_window_images,dtype=np.float32)
			# print (type(all_window_images))
			# all_window_images = torch.from_numpy(all_window_images)
			print ("here")
			sys.exit()
		


def draw_boxes(boxes,image):
	draw = ImageDraw.Draw(source_img)
	for category, box in boxes:
		draw.rectangle(((box[0], box[1]), (box[2], box[3])),outline='red')
		draw.text((box[0], box[1]), category)
		source_img.save('Data/Result/' + image, "JPEG")


# def aegon_targaryen_non_maximum_supression(boxes,threshold = 0.3):
	


resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 21)
resnet18.load_state_dict(torch.load(model_file_resnet))  
test2(resnet18)
# print (len(train_loader))
# print (count)
# print(count2)

