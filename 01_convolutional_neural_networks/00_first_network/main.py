#!/usr/bin/env python3

# import the nn
from network_class import *

# main function to run
def main():
	n_classes = 20 # example number of classes
	net = Net(n_classes)
	print(net)


if __name__ == '__main__':
	main()