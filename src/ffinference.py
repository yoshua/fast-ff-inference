#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Imports
#

import ast
import cPickle                              as pkl
import cStringIO
import cv2
import getopt
import gzip
import h5py                                 as H
import inspect
import io
import math
import numpy                                as np
import os
import pdb
import scipy.linalg
import sys
import tarfile
import theano                               as T
import theano.tensor                        as TT
import theano.tensor.nnet                   as TTN
import theano.tensor.nnet.conv              as TTNC
import theano.tensor.nnet.bn                as TTNB
import theano.tensor.signal.pool            as TTSP
from   theano import config                 as TC
import theano.printing                      as TP
import time
import traceback




###############################################################################
# Utilities
#

def loadMNIST(path='mnist.pkl.gz'):
	#Load MNIST data
	f=gzip.open(path)
	data=pkl.load(f)
	f.close()
	
	#Pick apart the dataset
	mnist_train_data   = data[0][0]
	mnist_train_labels = data[0][1]
	mnist_valid_data   = data[1][0]
	mnist_valid_labels = data[1][1]
	mnist_test_data    = data[2][0]
	mnist_test_labels  = data[2][1]
	
	#Return what we want
	trainX, trainY = mnist_train_data, np.eye(10)[mnist_train_labels]
	validX, validY = mnist_valid_data, np.eye(10)[mnist_valid_labels]
	return trainX, trainY
def initWeights(r,c): # r=n, c=oldN
	scale = np.sqrt(2.0/(r+c))
	randM  = np.random.uniform(low=-scale, high=+scale, size=(r,c)).astype(TC.floatX)
	if True:
		U,s,VT = scipy.linalg.svd(randM, full_matrices=False)
		if r > c:
			return U
		else:
			return VT
	else:
		return randM
def initBiases(r):
	return np.zeros(shape=(r,), dtype=TC.floatX)

###############################################################################
# Implementation
#

class DBM(object):
	def __init__(self, config=[784,784,784,784], hLrn=0.01, hMom=0.9, hL2P=0.001):
		# Saved configuration
		self.config = config
		
		# Construct shared variables: Parameters and other controls
		self.constructShareds(float(hLrn), float(hMom), float(hL2P))
		
		# Construct Theano functions
		self.constructTheanoFuncs()
	def constructShareds(self, hLrn, hMom, hL2P):
		# Controls
		self.hLrn   = T.shared(np.full((), hLrn, dtype=TC.floatX)) # Learning Rate
		self.hMom   = T.shared(np.full((), hMom, dtype=TC.floatX)) # Momentum
		self.hL2P   = T.shared(np.full((), hL2P, dtype=TC.floatX)) # L2 norm Penalty
		self.hLin   = T.shared(np.full((), 0.0,  dtype=TC.floatX)) # Linearity (Initially 1.0)
		
		# Parameters
		self.pW = []
		self.pB = []
		self.vW = []
		self.vB = []
		
		oldN = self.config[0]
		self.pB.append(T.shared(initBiases(oldN), "pB0"))
		self.vB.append(T.shared(np.zeros_like(initBiases(oldN)), "vB0"))
		for i in xrange(1, len(self.config)):
			n    = self.config[i]
			pW   = T.shared(initWeights(n, oldN), "pW"+str(i-1))
			pB   = T.shared(initBiases(n), "pB"+str(i))
			vW   = T.shared(np.zeros_like(initWeights(n, oldN)), "vW"+str(i-1))
			vB   = T.shared(np.zeros_like(initBiases(n)), "vB"+str(i))
			self.pW.append(pW)
			self.pB.append(pB)
			self.vW.append(vW)
			self.vB.append(vB)
			oldN = n
	def constructTheanoFuncs(self):
		# Theano functions
		
		# ********** Create ff init function *****************
		self.h         = [TT.matrix()] # Input
		for i in xrange(1, len(self.config)):
			hm1     = self.h[i-1]      # Get reference to previous layers
			W       = self.pW[i-1]     # Matrix pW[i-1] relates the previous layer (i-1) to the
			                           # current one (i)
			B       = self.pB[i]       # Biases of current layer
			
			self.h += [TT.dot(self.rho(hm1), W.T) + B] # Append computation of next layer
		
		self.ffinitF          = T.function([self.h[0]], self.h)
		
		
		# ********** Create reconstruction penalty & training functions *****************
		# Copy list of hiddens h to h'
		self.hp  = self.h[:]
		
		# Reconstruction penalty and training functions
		self.reconstructLossFs = []
		self.trainFs           = []
		for i in xrange(len(self.config)-1):
			# Compute reconstruction hp
			h   = self.hp[i]
			B   = self.pB[i]
			Bp  = self.pB[i+1]
			W   = self.pW[i]
			
			self.hp[i] = B + TT.dot(self.rho(Bp + (TT.dot(self.rho(h), W.T))), W)
			
			# Compute squared-error reconstruction error between each h and its hp
			sqdiff     = (self.h[i]-self.hp[i])**2
			loss       = TT.sum(TT.mean(sqdiff, axis=1))
			
			# Add to list of reconstruction functions
			self.reconstructLossFs.append(T.function([self.h[0]], [loss]))
			self.trainFs.append(T.function([self.h[0]],
		                                   [loss],
		                                   updates=self.computeTrainUpdates(loss, i)))
		
		# ********** Create step function *****************
		# Create step function to report the magnitude of the step size at each resampling iteration.
		self.oldH = []
		for i in xrange(0, len(self.config)):
			self.oldH += [TT.matrix()]
		self.newH = self.oldH[:]
			
		# Resample odd layers
		for i in xrange(1, len(self.config), 2):
			if i == len(self.config)-1:
				# Last layer gets special treatment
				hm1 = self.newH[i-1]
				Wm  = self.pW[i-1]
				B   = self.pB[i]
				
				self.newH[i] = B + 1.0*(TT.dot(self.rho(hm1), Wm.T))
			else:
				hm1 = self.newH[i-1]
				hp1 = self.newH[i+1]
				Wm  = self.pW[i-1]
				Wp  = self.pW[i]
				B   = self.pB[i]
				
				self.newH[i] = B + 0.5*(TT.dot(self.rho(hm1), Wm.T) + TT.dot(self.rho(hp1), Wp))
		# Resample even layers
		for i in xrange(2, len(self.config), 2):
			if i == len(self.config)-1:
				# Last layer gets special treatment
				hm1 = self.oldH[i-1]
				Wm  = self.pW[i-1]
				B   = self.pB[i]
				
				self.newH[i] = B + 1.0*(TT.dot(self.rho(hm1), Wm.T))
			else:
				hm1 = self.newH[i-1]
				hp1 = self.newH[i+1]
				Wm  = self.pW[i-1]
				Wp  = self.pW[i]
				B   = self.pB[i]
				
				self.newH[i] = B + 0.5*(TT.dot(self.rho(hm1), Wm.T) + TT.dot(self.rho(hp1), Wp))
		
		# Compute squared-error reconstruction error between each h and its hp
		self.oldNewSqdiffs    = map(lambda x: TT.sum(TT.mean((x[0]-x[1])**2, axis=0)), zip(self.oldH, self.newH))
		self.oldNewLoss       = reduce(lambda a,b:a+b, self.oldNewSqdiffs)
		self.stepF            = T.function(self.oldH, self.newH + [self.oldNewLoss])
	def computeTrainUpdates(self, loss, i):
		updatesList = []
		
		for pW,vW in zip(self.pW, self.vW)[i:i+1]:
			# Gradients
			dpW = T.grad(loss, pW)
			
			# Update for weights parameter and velocity
			uvW = (vW, self.hMom*vW + (1-self.hMom)*self.hLrn*dpW)
			upW = (pW, pW - uvW[1])
			
			# Add to updates list
			updatesList += [uvW, upW]
		for pB,vB in zip(self.pB, self.vB)[i+(0 if i==0 else 1):i+2]:
			# Gradients
			dpB = T.grad(loss, pB)
			
			# Update for biases parameter and velocity
			uvB = (vB, self.hMom*vB + (1-self.hMom)*self.hLrn*dpB)
			upB = (pB, pB - uvB[1])
			
			# Add to updates list
			updatesList += [uvB, upB]
		
		return updatesList
	def rho(self, x):
		hsigmx = TT.clip(x, 0.0, 1.0)                          # Hard Sigmoid
		return self.hLin*x + (1.0-self.hLin)*hsigmx            # Blend between x and hsigm(x)
	def run(self, trainX, trainY):
		# Savefile
		f = H.File("save-nontrained-ortho-uniform-784.hdf5")
		
		# Training
		maxE = 50
		bS   = 50
		
		"""
		# For each layer:
		for l in xrange(len(self.config)-1):
			self.hLrn.set_value(np.full((), 0.01, dtype=TC.floatX))
			
			# For each batch:
			for e in xrange(maxE):
				# Set linearity parameter:
				self.hLrn.set_value(np.full((), self.hLrn.get_value()*0.97, dtype=TC.floatX))
				
				# Print epoch header
				sys.stdout.write("Layer {:2d}, Epoch {:2d}\n".format(l, e))
				sys.stdout.flush()
				
				# Train on each batch:
				for b in xrange(len(trainX)/bS):
					batch = trainX[b*bS:(b+1)*bS]
					loss  = self.trainFs[l](batch)[0][()]
					sys.stdout.write("{:20.17f}\r".format(loss))
					sys.stdout.flush()
				
				# Print epoch trailer
				sys.stdout.write("\n")
				sys.stdout.flush()
		"""
		
		#Save weights
		for p in self.pW+self.pB:
			try:
				f.create_dataset("params/"+p.name, data=p.get_value())
			except:
				print p.name
				quit()
		
		# Now, use it.
		TEST_STEPS = 100
		logD = f.create_dataset("log/stepsize", shape=(TEST_STEPS,), maxshape=(None,), dtype=TC.floatX)
		h = self.ffinitF(trainX[0:50])
		#pdb.set_trace()
		for i in xrange(TEST_STEPS):
			ret = self.stepF(*h)
			h = ret[:-1]
			logD[i] = ret[-1][()]
			print ret[-1][()]


###############################################################################
# Implementations of the script's "verbs".
#

def verb_help(argv=None):
	print """Run this script with a verb argument, like "train"."""
def verb_train(argv=None):
	"""Train."""
	
	dbm = DBM()
	dbm.run(*loadMNIST())
def verb_screw(argv=None):
	"""Screw around."""
	
	pass
def verb_dumpimage(arg=None):
	import pylab
	import matplotlib.pyplot as plt
	
	fNTNO = H.File("save-nontrained-nonortho.hdf5", "r")
	fNTO  = H.File("save-nontrained-ortho.hdf5", "r")
	fTO   = H.File("save-trained-ortho.hdf5", "r")
	fTNO  = H.File("save-trained-nonortho.hdf5", "r")
	
	fig   = plt.figure(figsize=(10,5))
	ax    = fig.add_subplot(1,1,1)
	
	ax.set_title("Relaxation Step Size")
	ax.set_xlabel("Step #")
	ax.set_ylabel("L2 Norm of Step Size")
	ax.tick_params(which="both", top="off", right="off")
	ax.set_xlim([0, 100])
	
	ax.plot(xrange(1,101), fNTNO["/log/stepsize"][:], linewidth=2, color="red",   label="Random")
	ax.plot(xrange(1,101), fTO  ["/log/stepsize"][:], linewidth=2, color="blue",  label="Auto-Encoder")
	
	ax.legend(loc="upper right")
	
	pylab.savefig("plotstepsize.png")
	ax.semilogy()
	pylab.savefig("plotstepsizelog.png")


###############################################################################
# Main
#

if __name__ == "__main__":
    #
    # This script is invoked using a verb that denotes the general action to
    # take, plus optional arguments. If no verb is provided or the one
    # provided does not match anything, print a help message.
    #
    
    if((len(sys.argv) > 1)                      and # Have a verb?
       ("verb_"+sys.argv[1] in globals())       and # Have symbol w/ this name?
       (callable(eval("verb_"+sys.argv[1])))):      # And it is callable?
        eval("verb_"+sys.argv[1])(sys.argv)         # Then call it.
    else:
        verb_help(sys.argv)                         # Or offer help.
