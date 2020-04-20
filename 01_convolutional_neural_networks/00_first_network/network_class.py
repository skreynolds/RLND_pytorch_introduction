import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):


	def __init__(self, n_classes):
		super(Net, self).__init__()

		# 1 input image channel (grayscale), 32 output channels/feature maps
		# 5x5 square convolutional kernel
		self.conv1 = nn.Conv2d(1, 32, 5)

		# maxpool layer
		# pool with kernel_size=2, stride=2
		self.pool = nn.MaxPool2d(2, 2)

		# fully connected layer
		# 32*4 input size to account for the downsampled image size after pooling
		self.fcl = nn.Linear(32*4, n_classes)


	# define feedforward behaviour
	def forward(self, x):
		# one conv/relu + pool layers
		x = self.pool(F.relu(self.conv1(x)))

		# prep for linear layer by flattening the feature maps into feature vectors
		x = x.view(x.size(0), -1)
		# linear layer
		x = F.relu(self.fcl(x))

		# final output
		return x