import numpy as np
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import random 

def prediction(X,WL1,WL2,L1b,L2b,Y):
	
	for i in range(100):
		l = np.array([X[i]])
		L1Z = forwar_Pass(l.T, WL1, L1b)
		L1A = sigmoid(L1Z)
		L2Z = forwar_Pass(L1A, WL2, L2b)
		activation = sigmoid(L2Z)
		label = 0 if activation <0.5 else 1
		print("activation={}; predicted_label={}, true_label={}".format(activation, label, Y[i]))

 def sigmoid(Z):
	return (1 / (1 + np.exp(-Z)))
 def tanh(Z):
	return
def relu():
	return
def sigmoid_derivative(X):
	return sigmoid(X) * (1 - sigmoid(X))
def forwar_Pass(X,W,b):
	return np.dot(W,X) + b
def derivative_w_r_t_to_weights(dz, A, m):
	return (np.dot(dz,A) / m )
def Getting_Data_Set_Ready():
	b1 = b2 = 1
	split = 150
	dataset = np.genfromtxt('BinaryDataSet.csv',delimiter = ' ')
	np.random.shuffle(data_set)
	print("Shape of the data is ",data_set.shape)
	train,test = dataset[:split,:],data_set[split:,:]

	W1 = np.random.rand(100,324)
	W2 = np.random.rand(1,100)
	X,Y = train[:,:-1],train[:,-1]
	X = X.T 
	Y = np.array([Y])

	print("shape of X ",X.shape)
	print("Shape of Y ",Y.shape)
	print("shape of Layer 1 weights ",W1.shape)
	print("Shape of Layer 2 weights ",W2.shape)
	return X,Y,test,W1,W2,b1,b2

def Neural_Network():
	X,Y,test,WL1,WL2,L1b,L2b = Getting_Data_Set_Ready()
	print(WL2.shape)
	alpha = 0.0001
	m = np.prod(X.shape)

	for i in range(450):
		L1Z = forwar_Pass(X, WL1, L1b)
		L1A = sigmoid(L1Z)
		L2Z = forwar_Pass(L1A, WL2, L2b)
		L2A = sigmoid(L2Z)

		#backward Pass
		
		L2dz = L2A - Y

		L2dw = derivative_w_r_t_to_weights(L2dz, L1A.T,m)
		L2db = np.sum(L2dz, axis = 1, keepdims = True)
		L1dz = np.dot(W_L_2.T,L_2_dz) * sigmoid_derivativa(L1Z)
		L1dw = derivative_w_r_t_to_weights(L1dz, X.T, m)
		L1db = (np.sum(L1dz, axis = 1, keepdims = True))
		error1 = np.sum(L1dz**2)
		error2 = np.sum(L2dz**2)
		WL1 -=  0.01* WL1
		WL2 -=  0.01* WL2

		L1b -= 0.01 * L1b
		L2b -= 0.01 * L2b
		


	NewX, NewY = test[:,:-1],test[:,-1]
	NewY = np.array([NewY])
	print("W1",WL1.shape)
	print("X ",NewX.shape)
	print("Y ",NewY.shape)
	prediction(NewX,WL1,WL2,L1b,L2b,NewY.T)
Neural_Network()
