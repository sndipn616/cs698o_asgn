We could not use Jupyter Notebook with pytorch so we are submitting .py files.
1. assignment_1_1.py - Completed code for Part-1
2. assignment_1_2.py - Completed code for Part-2

The above codes will only run on a GPU. Furthermore the GPU I was using had some problem with torchvision.transforms and it was throwing error when scaling. Thats why we used our computer to dump the image in a binary format on our own machine after scaling and later used it on GPU to train and test the models.

The testing accuracy is in the following files. These files are written directly through the program and are not hand-written. The evaluator can find file writing code in the .py files.
1. output_1_1.txt - For Part-1
2. output_1_2.txt - For Part-2

Brief description of results-
1. For Part-1, we coded and evaluated for both VGG16 and RESNET18.
	For both the models we varied the learning rate as 0.1, 0.01, 0.001, 0.0001, 0.00001

	For VGG16 the accuracy was very low for learning rate = 0.1, 0.01 and 0.001 at about 10% which is as good as random class assignment.
	For learning rate = 0.0001 the accuracy reaches the highest value at 95.93%.
	For learning rate = 0.00001, the accuracy slightly decreases to 95.40%.

	For RESNET18 the trend was opposite.
	For learning rate = 0.1 the accuracy reaches the highest value at 89.30%.
	For learning rate = 0.01 the accuracy slightly decreases to 87.70%.
	For lower values of learning rate the accuracy decreases very much to 44% and then 36%.

2. For the custom RESNET model we varied the learning rate as 0.1, 0.01, 0.001, 0.0001, 0.00001
	For learning rate = 0.1 the accuracy was 79.79%.
	For learning rate = 0.01 the accuary increases to 91.76%.
	For learning rate = 0.001 and 0.0001, the accuracy further increases to 92.13% and 92.94% respectively.
	For learning rate = 0.00001, the accuracy decreases again to 89.68%.

The plots of loss vs iteration is present in plots directory. The plots are properly labelled for anyone to understand.
