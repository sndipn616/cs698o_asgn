
# coding: utf-8

# ## Creating Custom Networks
# In this notebook you have to create a custom network whose architecture has been given, and use the dataset you created earlier to train and test it.

# In[ ]:


# Import Statements
#
# Several of the imports you will need have been added but you will need to provide the
# rest yourself; you should be able to figure out most of the imports as you go through
# the notebook since without proper imports your code will fail to run
#
# All import statements go in this block

from __future__ import division, print_function, unicode_literals
import os
import sys
import numpy as np
import torch
import cPickle
import torch.nn as nn
import torch.utils.data
from PIL import Image
from scipy.ndimage import imread
from collections import OrderedDict
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# All hyper parameters go in the next block

# In[ ]:
# torch.cuda.device(1)

with torch.cuda.device(2):

    root_dir = 'notMNIST_small'
    batch_size = 10
    num_epochs = 5
    learning_rate = 0.01
    numClasses = 10
    use_gpu = True
    #model_file = 'custom_resnet_trained'
    model_file_resnet = 'custom_resnet'
    cifar_100 = 'cifar-100-python/test'


    # ### Create Custom Dataset and Loader
    # This is the same as part 1. Simply use the same code to create the dataset.

    # In[ ]:


    class CDATA(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
        def __init__(self, root_dir, train, transform=None):
            # root_dir  - the root directory of the dataset
            # train     - a boolean parameter representing whether to return the training set or the test set
            # transform - the transforms to be applied on the images before returning them
            #
            # In this function store the parameters in instance variables and make a mapping
            # from images to labels and keep it as an instance variable. Make sure to check which
            # dataset is required; train or test; and create the mapping accordingly.
            self.train = train
            self.root_dir = root_dir
            self.transform = transform
            self.training_folder = 'train'
            self.test_folder = 'test'
            self.training_file = 'training2.pt'
            self.test_file = 'test2.pt'
            self.processed_folder = 'processed'

            if not os.path.exists(os.path.join(self.root_dir, self.processed_folder, self.training_file)) and not os.path.exists(os.path.join(self.root_dir, self.processed_folder, self.test_file)):
                self.Process_Dataset()

            if self.train:
                self.train_data, self.train_labels = torch.load(
                    os.path.join(self.root_dir, self.processed_folder, self.training_file))
            else:
                self.test_data, self.test_labels = torch.load(os.path.join(self.root_dir, self.processed_folder, self.test_file))
            
        def __len__(self):
            # return the size of the dataset (total number of images) as an integer
            # this should be rather easy if you created a mapping in __init__
            if self.train:
                return len(self.train_data)
            else:
                return len(self.test_data)
            
        def __getitem__(self, idx):
            # idx - the index of the sample requested
            #
            # Open the image correspoding to idx, apply transforms on it and return a tuple (image, label)
            # where label is an integer from 0-9 (since notMNIST has 10 classes)
            if self.train:
                img, target = self.train_data[idx], self.train_labels[idx]
            else:
                img, target = self.test_data[idx], self.test_labels[idx]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            # print (img)
            # img = Image.fromarray(img.numpy(), mode='L')

            # if self.transform is not None:
            #     img = self.transform(img)        

            return img, target

        def GetChar(self, direc):
           
            if direc[0] == 'A':
                return 0
            elif direc[0] == 'B':
                return 1
            elif direc[0] == 'C':
                return 2
            elif direc[0] == 'D':
                return 3
            elif direc[0] == 'E':
                return 4
            elif direc[0] == 'F':
                return 5
            elif direc[0] == 'G':
                return 6
            elif direc[0] == 'H':
                return 7
            elif direc[0] == 'I':
                return 8
            elif direc[0] == 'J':
                return 9

            
                

        def Process_Dataset(self):
            if not os.path.exists(os.path.join(self.root_dir, self.processed_folder)):
                os.makedirs(os.path.join(self.root_dir, self.processed_folder))    #Create new folder

            train_images = []
            train_labels = []
            test_images = []
            test_labels = []

            for direc in os.listdir(os.path.join(self.root_dir, self.training_folder)):
                for img in os.listdir(os.path.join(self.root_dir, self.training_folder,direc)):
                    image = imread(os.path.join(self.root_dir, self.training_folder, direc, img))
                    image = torch.from_numpy(image)
                    image = Image.fromarray(image.numpy(), mode='L')

                    if self.transform is not None:
                        image = self.transform(image)

                    train_images.append(image) 
                    
                    # print (type(temp))                           
                    train_labels.append(self.GetChar(direc))
                    # image.close()
            train_labels = np.array(train_labels)
            train_labels = torch.from_numpy(train_labels)
            # print (len(train_labels[1]))
            # print (np.array(train_labels).shape)
            # train_labels = torch.LongTensor(train_labels)
            training_set = (train_images,train_labels)

            for direc in os.listdir(os.path.join(self.root_dir, self.test_folder)):
                for img in os.listdir(os.path.join(self.root_dir, self.test_folder,direc)):
                    image = imread(os.path.join(self.root_dir, self.test_folder, direc, img))
                    image = torch.from_numpy(image)
                    image = Image.fromarray(image.numpy(), mode='L')

                    if self.transform is not None:
                        image = self.transform(image)

                    test_images.append(image)                            
                    test_labels.append(self.GetChar(direc))
                    

                    # image.close()
            test_labels = np.array(test_labels)
            test_labels = torch.from_numpy(test_labels)
            # print (test_labels)
            # test_labels = torch.LongTensor(test_labels)
            test_set = (test_images, test_labels)

            with open(os.path.join(self.root_dir, self.processed_folder, self.training_file), 'wb') as f:
                torch.save(training_set, f)
            with open(os.path.join(self.root_dir, self.processed_folder, self.test_file), 'wb') as f:
                torch.save(test_set, f)


        
    composed_transform = transforms.Compose([transforms.Scale((32,32)),transforms.ToTensor()])
    train_dataset = CDATA(root_dir=root_dir, train=True, transform=composed_transform) # Supply proper root_dir
    test_dataset = CDATA(root_dir=root_dir, train=False, transform=composed_transform) # Supply proper root_dir

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    # ### Creating a Custom Network
    # It's time to create a new custom network. This network is based on Resnet (indeed it is a resnet since it uses skip connections). The architecture of the network is provided in the diagram. It specifies the layer names, layer types as well as their parameters.
    # <img src="architecture.png" width=100>
    # [Full size image](architecture.html)

    # In[ ]:

    class CustomResnet(nn.Module): # Extend PyTorch's Module class
        def __init__(self, num_classes = 10):
            super(CustomResnet, self).__init__() # Must call super __init__()
            
            # Define the layers of the network here
            # There should be 17 total layers as evident from the diagram
            # The parameters and names for the layers are provided in the diagram
            # The variable names have to be the same as the ones in the diagram
            # Otherwise, the weights will not load
            self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=True)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3,2,1)

             
            self.lyr1conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True) 
            self.lyr1bn1 = nn.BatchNorm2d(64)
            self.lyr1relu1 = nn.ReLU(inplace=True)
            self.lyr1conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True) 
            self.lyr1bn2 = nn.BatchNorm2d(64)

            self.lyr1relu2 = nn.ReLU(inplace=True)

            self.lyr2conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True) 
            self.lyr2bn1 = nn.BatchNorm2d(64)
            self.lyr2relu1 = nn.ReLU(inplace=True)
            self.lyr2conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True) 
            self.lyr2bn2 = nn.BatchNorm2d(64)

            self.lyr2relu2 = nn.ReLU(inplace=True)

            self.fc = nn.Linear(4096, num_classes)  

            

            
        def forward(self, x):
            # Here you have to define the forward pass
            # Make sure you take care of the skip connections
            # print (type(x))
            # x = torch.ByteTensor(x)
            # print (x.size())
            out = self.conv1(x)
            # print (out.size())
            out = self.bn1(out)
            # print (out.size())
            self.relu(out)
            # print (out.size())
            out = self.maxpool(out)
            # print (out.size())      

            out1 = self.lyr1conv1(out)
            # print (out1.size())
            out1 = self.lyr1bn1(out1)
            # print (out1.size())
            self.lyr1relu1(out1)
            # print (out1.size())
            out1 = self.lyr1conv2(out1)
            # print (out1.size())
            out1 = self.lyr1bn2(out1)
            # print (out1.size())  

            out = out + out1
            # print (out.size())
            self.lyr1relu2(out)
            # print (out.size())

            out1 = self.lyr2conv1(out)
            # print (out1.size())
            out1 = self.lyr2bn1(out1)
            # print (out1.size())
            self.lyr2relu1(out1)
            # print (out1.size())
            out1 = self.lyr2conv2(out1)
            # print (out1.size())
            out1 = self.lyr2bn2(out1)
            # print (out1.size())

            out = out + out1
            # print (out.size())
            self.lyr2relu2(out)

            # print (out.size())

            out = out.view(out.size(0), -1)
            # print (out.size())
            out = self.fc(out)
            
            # print (out.size())
            return out


    # #### Finetune on pre-trained CIFAR-100 weights
    # We shall now finetune our model using pretrained CIFAR-100 weights.

    # In[ ]:
    def unpickle(file):
        
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict


    model = CustomResnet(num_classes = 100) # 100 classes since CIFAR-100 has 100 classes

    # Load CIFAR-100 weights. (Download them from assignment page)
    # If network was properly implemented, weights should load without any problems
    model.load_state_dict(torch.load('CIFAR-100_weights')) # Supply the path to the weight file


    # ##### Optional
    # As a sanity check you may load the CIFAR-100 test dataset and test the above model. You should get an accuracy of ~41%. This part is optional and is meant for your convenience.

    # In[ ]:


    # Block for optionally running the model on CIFAR-100 test set
    print ("Doing Sanity Check")

    def sanity_test(model, features, labels, total_images):
        # Write loops for testing the model on the test set
        # You should also print out the accuracy of the model
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(np.array(labels))
        i = 0
        correct = 0
        while i < total_images:
            current_features = Variable(features[i:i+batch_size])
            current_labels = labels[i:i+batch_size]

            outputs = model(current_features)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted.cpu() == current_labels.cpu()).sum()

            i += batch_size

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total_images))
        return (100 * correct / total_images)

       

    data_dict = unpickle(cifar_100)
    total_images = len(data_dict['fine_labels'])
    features = data_dict['data'].reshape(total_images, 3, 32, 32)
    labels = data_dict['fine_labels']
    # print (feature.shape)
    sanity_test(model, features, labels, total_images)




    # Let's finetune the model.

    # In[ ]:


    # Change last layer to output 10 classes since our dataset has 10 classes
    model.fc = nn.Linear(model.fc.in_features, numClasses) # Complete this statement. It is similar to the resnet18 case

    # Loss function and optimizers
    criterion = nn.CrossEntropyLoss() # Define cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) # Use Adam optimizer, use learning_rate hyper parameter

    if use_gpu:
        model.cuda()

    def train(file):
        # Code for training the model
        # Make sure to output a matplotlib graph of training losses
        print ("Training Resnet")
        model_file_resnet = file
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):  
                # Convert torch tensor to Variable           
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
                # if(use_gpu):
                #     images=images.cuda()
                #     labels=labels.cuda()
                # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer
                images = torch.cat((images, images, images), 1)
                outputs = model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # sys.exit()
                if (i+1) % 10 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                           %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

        torch.save(model.state_dict(), model_file_resnet)

    # get_ipython().magic(u'time train()')


    # Test the finetuned model

    # In[ ]:


    def test(model):
        # Write loops for testing the model on the test set
        # You should also print out the accuracy of the model
        correct = 0
        total = 0
        print ("Testing")
        for images, labels in test_loader:        
            
            if(use_gpu):
                images = Variable(images.cuda())
            # else:
            #     images = Variable(images)
                
            images = torch.cat((images, images, images), 1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        return (100 * correct / total)
        
    # get_ipython().magic(u'time test()')
    
    f = open('result_part2.txt', 'w')
    train('custom_resnet_model')
    acc = test(model)
    f.write("Accuracy of Custom RESNET with learning rate = 0.01 " + str(acc)+'\n')

    # Reinstantiate the model and optimizer
    #model = CustomResnet(num_classes = 10)
    #optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)# Use Adam optimizer, use learning_rate hyper parameter
   # model.load_state_dict(torch.load('CIFAR-100_weights'))

    #model.cuda()

    #train('custom_resnet_type-2')
    #acc = test(model)
    #f.write(str(acc)+'\n')
    f.close()
    
    '''
    # Train
    # get_ipython().magic(u'time train()')

    # Test
    # get_ipython().magic(u'time test()')


# This is the end of Assignment 1
'''
