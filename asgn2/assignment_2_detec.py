from __future__ import division, print_function, unicode_literals
import os
import sys
import torch
import numpy as np
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
import xml.etree.cElementTree as ET

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


resnet_input = 224  #size of resnet18 input images
back_patch_size = 64

window_size = [32, 64, 96, 128]
aspect_ratio = [1, 2, 4]
stride = 2

# Cersei chose violence, you choose your hyper-parameters wisely using validation data!
batch_size = 5
num_epochs = 2
learning_rate_list =  [0.1, 0.01, 0.001, 0.0001, 0.00001]
hyp_momentum = 0.9
dataset_size = 5000
root_dir = 'Data'
result_dir = 'Result'
back_class = '__background__'
result_file = 'result_classification.txt'
resnet18_loss_file = 'resnet18_loss'
model_file_resnet = 'resnet18_model'

# # Testing and Accuracy Calculation
# Jorah then asks a question, how is this a detection task?<br/>
# As everybody wonders, Theon Greyjoy suggests a slding window method to test the above trained trained network on the detection task:<br/>
# "We take some windows of varying size and aspect ratios", he mumbled, "and slide it through the test image (considering some stride of pixels) from left to right, and top to bottom, detect the class scores for each of the window, and keep only those which are above a certain threshold value!". "He is right", says Samwell, "I read a similar approach in the paper -Faster RCNN by Ross Girshick in the library, where he uses three diferent scales/sizes and three different aspect ratios, making a total of nine windows per pixel to slide". You need to write the code and use it in testing code to find the predicted boxes and their classes.

# In[ ]:

composed_transform = transforms.Compose([transforms.ToTensor()])

test_dataset = hound_dataset(root_dir=root_dir, train=False, transform=composed_transform) # Supply proper root_dir
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

def theon_sliding_window(model):
	print ("Applying Sliding Window")
	directory_name = 'result'
	index = 0
	for ws in window_size:
		for ar in aspect_ratio:
			i = 0
			while i < 2:
				if i == 0:
					window_width = wr
					window_height = wr*ar

				else:
					window_height = wr
					window_width = wr*ar

				current_directory = result_dir + directory_name + str(ws) + str(ar) + str(i)
				if not os.path.exists(current_directory):
    				os.makedirs(current_directory)

			    for images, labels in test_loader:
			        # images = Variable(images)  
			        img = Image.fromarray(images.numpy(), mode='RGB')
			        img.save("img" + str(index), "JPEG")
			        
			        root = ET.Element("root")
			        boxes = ET.SubElement(root, "boxes")

			        img_channel, img_width, img_height = img.size
			        
			        ymin = 0
			        while xmax < img_width:
			        	xmin = 0
			        	xmax = xmin + window_width
			        	ymax = ymin + window_height

			        	while ymax < img_height:
				        	window_image = img.crop((xmin,ymin,xmax,ymax))
				        	window_image = window_image.resize((224,224), Image.BILINEAR)
				        	window_image = composed_transform(window_image)

				        	output = model(window_image)
				        	_, predicted = torch.max(outputs.data, 1)

				        	if predicted != 0:
				        		box = ET.SubElement(boxes, "box")
				        		box.text = predicted
				        		ET.SubElement(box, "xmin").text = str(xmin)
				        		ET.SubElement(box, "ymin").text = str(ymin)
				        		ET.SubElement(box, "xmax").text = str(xmax)
				        		ET.SubElement(box, "ymax").text = str(ymax)


				        	xmin = xmin + stride
				        	xmax = xmin + window_width
				        
				        ymin = ymin + stride
				        ymax = ymin + window_height	

				 	tree = ET.ElementTree(root)
					tree.write("img" + str(index) + "_raw.xml")
				 	index += 1

			    i += 1
			    if ar == 1:
			    	break   
    


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

resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 21)
resnet18.load_state_dict(torch.load(model_file_resnet))
theon_sliding_window(resnet18)
daenerys_test(resnet18) 

# # Final Showdown
# After covering all the steps and passing the accuracy value to the talking crystal, they all pass through to the land of the living, with a wounded Jon Snow armed with the Dragon-axe. After a fierce battle, Jon Snow manages to go face to face with the Night king. Surrounded by battling men and falling bodies, they engage in a ferocious battle, a battle of spear and axe. After a raging fight, Jon manages to sink the axe into the Night king's heart, but not before he gets wounded by the spear. As dead men fall to bones, Daenerys and others rush to his aid, but it is too late. Everyone is in tears as they look towards the man of honour, Jon Snow, lying in Daenerys's arms when he says his last words: "The night has ended. Winter is finally over!"

# In[ ]: