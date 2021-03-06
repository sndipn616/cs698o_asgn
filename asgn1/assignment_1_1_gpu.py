
# coding: utf-8

# ## Creating Custom Datasets and Finetuning Pre-trained Networks
# In this notebook you have to create custom datasets for PyTorch and use this dataset to finetune certain pre-trained neural networks and observe the results.

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
import torch
import numpy as np
import torchvision
import unicodedata
import cPickle as pkl
from PIL import Image
import torch.nn as nn
import torch.utils.data
from scipy.ndimage import imread
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# All hyper parameters go in the next block

# In[ ]:

with torch.cuda.device(1):


    root_dir = 'notMNIST_small'
    batch_size = 5
    num_epochs = 5
    learning_rate_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    num_classes = 10
    use_gpu = True
    model_file_vgg = 'vgg16_model'
    model_file_resnet = 'resnet18_model'
    vgg16_loss_file = 'vgg16_loss'
    resnet18_loss_file = 'resnet18_loss'
    result_file = 'result.txt'

    # ### Creating Custom Datasets
    # Your first task is to create a pipeline for the custom dataset so that you can load it using a dataloader. Download the dataset provided in the assignment webpage and complete the following block of code so that you can load it as if it was a standard dataset.

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
            self.training_file = 'training.pt'
            self.test_file = 'test.pt'
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


    # We shall now load the dataset. You just need to supply the `root_dir` in the block below and if you implemented the above block correctly, it should work without any issues.

    # In[ ]:


    composed_transform = transforms.Compose([transforms.Scale((224,224)),transforms.ToTensor()])
    # composed_transform = transforms.Compose([transforms.Scale((56,56)),transforms.ToTensor()])

    train_dataset = CDATA(root_dir=root_dir, train=True, transform=composed_transform) # Supply proper root_dir
    test_dataset = CDATA(root_dir=root_dir, train=False, transform=composed_transform) # Supply proper root_dir

    # Let's check the size of the datasets, if implemented correctly they should be 16854 and 1870 respectively
    print('Size of train dataset: %d' % len(train_dataset))
    print('Size of test dataset: %d' % len(test_dataset))

    # Create loaders for the dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Let's look at one batch of train and test images
    def imshow(img):
        npimg = img.numpy()    
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        
    # train_dataiter = iter(train_loader)
    # train_images, train_labels = train_dataiter.next()
    # print("Train images")
    # imshow(torchvision.utils.make_grid(train_images))


    # test_dataiter = iter(test_loader)
    # test_images, test_labels = test_dataiter.next()
    # print("Test images")
    # imshow(torchvision.utils.make_grid(test_images))
    # print(test_images[0].shape)
    # print(test_labels)
    # print(type(test_images))
    # print(type(test_labels))



    # ### VGG-16 and Resnet-18
    # Now that you have created the dataset we can use it for training and testing neural networks. VGG-16 and Resnet-18 are both well-known deep-net architectures. VGG-16 is named as such since it has 16 layers in total (13 convolution and 3 fully-connected). Resnet-18 on the other hand is a Resnet architecture that uses skip-connections. PyTorch provides pre-trained models of both these architectures and we shall be using them directly. If you are interested in knowing how they have been defined do take a look at the source, [VGG](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py), [Resnet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

    # In[ ]:


    def GetModels():
        vgg16 = models.vgg16(pretrained=True)
        resnet18 = models.resnet18(pretrained=True)

        for param in vgg16.parameters():
            param.requires_grad = True

        for param in resnet18.parameters():
            param.requires_grad = True

        # Code to change the last layers so that they only have 10 classes as output
        vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

        resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)

        # Add code for using CUDA here if it is available
        if use_gpu:
            print ("Converting models for GPU")
            vgg16.cuda()
            resnet18.cuda()

        return vgg16, resnet18

    # vgg16.load_state_dict(torch.load(model_file_vgg))
    # resnet18.load_state_dict(torch.load(model_file_resnet))
    # Define loss functions and optimizers

    # In[ ]:
    

    # #### Finetuning
    # Finetuning is nothing but training models after their weights have been loaded. This allows us to start at a better position than training from scratch. Since the models created already have weights loaded, you simply need to write a training loop.

    # In[ ]:

    def save_loss(loss,filename):
	with open(filename, "wb") as fp:
		pkl.dump(loss,fp)

    def get_loss(filename):
	with open(filename, "rb") as fp:
		temp = pkl.load(fp)
		return temp 

    def train_vgg16(sl):
        # Write loops so as to train the model for N epochs, use num_epochs hyper parameter
        # Train/finetune the VGG-16 network
        # Store the losses for every epoch and generate a graph using matplotlib
        # print ("Here")
        print ("Training VGG")
        loss_vgg16 = []
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):  
                # Convert torch tensor to Variable           
                #images_ = Variable(images)
                #labels = Variable(labels)
                if(use_gpu):
                    images=Variable(images.cuda())
                    labels=Variable(labels.cuda())
#    	    	else:
 #   		    images=Variable(images)
  #  		    labels=Variable(labels)

                # Forward + Backward + Optimize
                optimizer_vgg16.zero_grad()  # zero the gradient buffer
                images = torch.cat((images, images, images), 1)
                outputs = vgg16(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_vgg16.step()
                # sys.exit()
                if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                           %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
		    
                    loss_vgg16.append(loss.data[0])

        torch.save(vgg16.state_dict(), model_file_vgg + str(sl))
        save_loss(loss_vgg16,vgg16_loss_file + str(sl) + ".txt")
                # if i == 2000:
                #     break
       
    def train_resnet18(sl):
        # Same as above except now using the Resnet-18 network
        print ("Training Resnet")
        loss_resnet18 = []
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):  
                # Convert torch tensor to Variable           
                #images_ = Variable(images)
                #labels = Variable(labels)
                if(use_gpu):
                    images=Variable(images.cuda())
                    labels=Variable(labels.cuda())
                # Forward + Backward + Optimize
                optimizer_resnet18.zero_grad()  # zero the gradient buffer
                images = torch.cat((images, images, images), 1)
                outputs = resnet18(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_resnet18.step()
                # sys.exit()
                if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                           %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
                    loss_resnet18.append(loss.data[0])

        torch.save(resnet18.state_dict(), model_file_resnet + str(sl))
        save_loss(loss_resnet18,resnet18_loss_file + str(sl) + ".txt")
                # if i == 2000:
                #     break


    # Now let us start the training/finetuning

    # In[ ]:

    '''
    get_ipython().magic(u'time train_vgg16()')
    get_ipython().magic(u'time train_resnet18()')


    # #### Testing
    # Once finetuning is done we need to test it on the test set.

    # In[ ]:
    '''

    def test(model):
        # Write loops for testing the model on the test set
        # You should also print out the accuracy of the model
        correct = 0
        total = 0
        print ("Testing")
        for images, labels in test_loader:
    #        images = Variable(images)
            
            if(use_gpu):
                images = Variable(images.cuda())
#    	else:
 #   	    images = Variable(images)
                
            images = torch.cat((images, images, images), 1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        return (100 * correct / total)


    # Test the models

    # In[ ]:

    '''
    get_ipython().magic(u'time test(vgg16)')
    get_ipython().magic(u'time test(resnet18)')


    # You can add more code to save the models if you want but otherwise this notebook is complete
    '''
    #criterion = nn.CrossEntropyLoss()   # Define cross-entropy loss
    #optimizer_vgg16 = torch.optim.Adam(vgg16.parameters(), lr = learning_rate / 10.0) # Use Adam optimizer, use learning_rate hyper parameter
    #optimizer_resnet18 = torch.optim.Adam(resnet18.parameters(), lr = learning_rate) # Use Adam optimizer, use learning_rate hyper parameter


    f = open(result_file, 'w')
    for learning_rate in learning_rate_list:
        vgg16, resnet18 = GetModels()
        criterion = nn.CrossEntropyLoss()   # Define cross-entropy loss
        optimizer_vgg16 = torch.optim.Adam(vgg16.parameters(), lr = learning_rate) # Use Adam optimizer, use learning_rate hyper parameter
        optimizer_resnet18 = torch.optim.Adam(resnet18.parameters(), lr = learning_rate) # Use Adam optimizer, use learning_rate hyper parameter

        train_vgg16(learning_rate)
        acc = test(vgg16)
        f.write("Accuracy of VGG16 with learning rate = " + str(learning_rate) + " : " + str(acc) + "\n")

        train_resnet18(learning_rate)
        acc = test(resnet18)
        f.write("Accuracy of RESNET18 with learning rate = " + str(learning_rate) + " : " + str(acc) + "\n")


    f.close()

    
    # train_vgg16(1)
    #vgg16.load_state_dict(torch.load(model_file_vgg+str(1)))
    # acc = test(vgg16)
    

    #train_resnet18()
    #optimizer_vgg16 = torch.optim.Adam(vgg16.parameters(), lr = learning_rate / 100.0)
    #num_epochs = 10
    #num_epochs = 1
    #train_vgg16(2)
    #vgg16.load_state_dict(torch.load(model_file_vgg+str(2)))
    #acc = test(vgg16)
    #f.write('lr=lr/100 ' + str(acc) + "\n")

    #optimizer_vgg16 = torch.optim.RMSprop(vgg16.parameters(), lr = learning_rate / 100)
    #train_vgg16(3)
    #vgg16.load_state_dict(torch.load(model_file_vgg+str(3)))
    #acc = test(vgg16)
    #f.write('rmsprop ' + str(acc) + '\n')

    #f = open('result_resnet.txt','w')
    #num_epochs = 5
    
    # f.write("Accuracy of RESNET18 with learning rate = 0.01 : " + str(acc) + '\n')

    
