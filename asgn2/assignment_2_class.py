from __future__ import division, print_function, unicode_literals
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

# ### Fine-tuning
# Litlefinger has brought you a pre-trained network. Fine-tune the network in the following section:

# In[ ]:
# You can ask Varys to get you more if you desire

resnet_input = 224  #size of resnet18 input images
back_patch_size = 64

# In[ ]:


# Cersei chose violence, you choose your hyper-parameters wisely using validation data!
batch_size = 5
num_epochs = 2
learning_rate_list =  [0.1, 0.01, 0.001, 0.0001, 0.00001]
hyp_momentum = 0.9
dataset_size = 5000
root_dir = 'Data'
back_class = '__background__'
result_file = 'result_classification.txt'
resnet18_loss_file = 'resnet18_loss'
model_file_resnet = 'resnet18_model'

# composed_transform = transforms.Compose([transforms.Scale((resnet_input, resnet_input)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
# composed_transform = transforms.Compose([transforms.Scale(resnet_input), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
composed_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])


train_dataset = hound_dataset(root_dir=root_dir, train=True, transform=composed_transform) # Supply proper root_dir
test_dataset = hound_dataset(root_dir=root_dir, train=False, transform=composed_transform) # Supply proper root_dir

# print('Size of train dataset: %d' % len(train_dataset))
# print('Size of test dataset: %d' % len(test_dataset))

# train_dataset = train_dataset[0:dataset_size]
# test_dataset = test_dataset[0:dataset_size]

# print('Size of train dataset: %d' % len(train_dataset))
# print('Size of test dataset: %d' % len(test_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


print('Size of train dataset: %d' % len(train_loader))
print('Size of test dataset: %d' % len(test_loader))

# train_loader = train_loader[0:dataset_size]
# test_loader = test_loader[0:dataset_size]

# print('Size of train dataset: %d' % len(train_loader))
# print('Size of test dataset: %d' % len(test_loader))

# resnet18 = models.resnet18(pretrained=True)

# resnet18.fc = nn.Linear(resnet18.fc.in_features, 21)

# Add code for using CUDA here


# In[ ]:


criterion = nn.CrossEntropyLoss()
# Update if any errors occur
# optimizer = optim.SGD(resnet18.parameters(), learning_rate, hyp_momentum)


# In[ ]:
def save_loss(loss,filename):
	with open(filename, "wb") as fp:
		pkl.dump(loss,fp)

def arya_train(sl,optimizer,criterion):
	# Begin
	print ("Training RESNET18")
	loss_resnet18 = []
	for epoch in range(num_epochs):
	    for i, (images, labels) in enumerate(train_loader):           
	        
	        images=Variable(images)
	        labels=Variable(labels)

	        # Forward + Backward + Optimize
	        optimizer.zero_grad()  # zero the gradient buffer
	        # images = torch.cat((images, images, images), 1)
	        outputs = resnet18(images)

	        loss = criterion(outputs, labels)
	        loss.backward()
	        optimizer.step()

	        if (i+1) % 100 == 0:
	            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
	                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
	    
	            loss_resnet18.append(loss.data[0])

	torch.save(resnet18.state_dict(), model_file_resnet + str(sl))
	save_loss(loss_resnet18,resnet18_loss_file + str(sl) + ".txt")

def test(model):
    # Write loops for testing the model on the test set
    # You should also print out the accuracy of the model
    correct = 0
    total = 0
    print ("Testing")
    for images, labels in test_loader:
        images = Variable(images)            
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).sum()
    print('Accuracy of the network after training on test dataset : %d %%' % (100 * correct / total))
    return (100 * correct / total)



print ("Starting Training")
f = open(result_file, 'w')
for learning_rate in learning_rate_list:        
	resnet18 = models.resnet18(pretrained=True)
	resnet18.fc = nn.Linear(resnet18.fc.in_features, 21)  

	optimizer_resnet = torch.optim.SGD(resnet18.parameters(), learning_rate, hyp_momentum) 
	arya_train(learning_rate,optimizer_resnet,criterion)
	acc = test(resnet18)
	f.write("Accuracy of RESNET18 with learning rate = " + str(learning_rate) + " : " + str(acc) + "\n")      


f.close()