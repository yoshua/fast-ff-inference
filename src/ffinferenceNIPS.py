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
# Needs:
#



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
def loadCIFAR10(path='cifar-10-batches-py/'):
	trainX = np.empty((0, 3072), dtype=TC.floatX)
	trainY = []
	
	for i in xrange(1,6):
		name    = path+"data_batch_"+str(i)
		f       = open(name, 'rb')
		dict    = pkl.load(f)
		f.close()
		
		bData   = dict["data"].astype(TC.floatX)/255.0
		trainX  = np.concatenate([trainX, bData])
		trainY += dict["labels"]
	
	trainY = np.eye(10)[trainY].astype(TC.floatX)
	
	return trainX, trainY
def initWeights(r,c,isOrtho=False): # r=n, c=oldN
	scale = np.sqrt(2.0/(r+c))
	randM  = np.random.uniform(low=-scale, high=+scale, size=(r,c)).astype(TC.floatX)
	if isOrtho:
		U,s,VT = scipy.linalg.svd(randM, full_matrices=False)
		if r > c:
			return U
		else:
			return VT
	else:
		return randM
def initBiases(r):
	return np.zeros(shape=(r,), dtype=TC.floatX)
def Adam(svParams, sv1stM, sv2ndM, vGrads, t, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
	"""Adam optimizer, from https://arxiv.org/pdf/1412.6980v8.pdf, Algo 1.
	Written for Theano.
	
	Alpha is a learning rate, which must be greater than 0. A sane default is 0.001.
	Beta1 is a decay rate for the 1st-order momentum estimate. The default is 0.9.
	Beta2 is a decay rate for the 2nd-order momentum estimate. The default is 0.999.
	"""
	
	# Scalar time-step update
	tNew = (t+1)
	updates = [(t, tNew.astype("uint64"))]
	
	tNew = tNew.astype(TC.floatX) # For use with floatX'es below
	
	# Parameter updates
	for (p, m1, m2, g) in zip(svParams, sv1stM, sv2ndM, vGrads):
		                                                  # Get gradients w.r.t. stochastic objective at timestep t.
		                                                  # (We rely on the caller to provide us this)
		m1New       = beta1 * m1  +  (1-beta1) * g        # Update biased first moment estimate
		m2New       = beta2 * m2  +  (1-beta2) * g * g    # Update biased second raw momentum estimate.
		m1Corrected = m1New / (1 - TT.pow(beta1, tNew))   # Compute bias-corrected first moment estimate.
		m2Corrected = m2New / (1 - TT.pow(beta2, tNew))   # Compute bias-corrected second raw moment estimate.
		pNew        = p -  alpha * m1Corrected / (        # Update parameters
		                   TT.sqrt(m2Corrected) + eps)
		
		updates += [(m1, m1New), (m2, m2New), (p, pNew)]
	
	# Return updates list.
	return updates
def zerosLikeSVs(svs, renamer=lambda x:None):
	return {x.name : T.shared(np.zeros_like(x.get_value()), renamer(x.name)) for x in svs.itervalues()}

###############################################################################
# Implementation
#

class GRN(object):
	def __init__(self, config=[784,1024,1024], hLrn=0.0001, hMom=0.9, hL2P=0.0001, isOrtho=False):
		self.hLrnBase = float(hLrn)
		self.hMomBase = float(hMom)
		self.hL2PBase = float(hL2P)
		
		self.config = config
		self.consSVs(self.hLrnBase, self.hMomBase, self.hL2PBase, isOrtho)
		self.consTFs()
	def consSVs(self, hLrn, hMom, hL2P, isOrtho=False):
		# Controls
		self.hLrn   = T.shared(np.full((), hLrn, dtype=TC.floatX)) # Learning Rate
		self.hMom   = T.shared(np.full((), hMom, dtype=TC.floatX)) # Momentum
		self.hL2P   = T.shared(np.full((), hL2P, dtype=TC.floatX)) # L2 norm Penalty
		
		# Parameters Construction
		self.U      = self.consUs(isOrtho) # Matrices of Forwards  Contribution
		self.V      = self.consVs(isOrtho) # Matrices of Backwards Contribution
		self.b      = self.consbs()        # Biases   on Forwards  Contribution
		self.c      = self.conscs()        # Biases   on Backwards Contribution
		
		# Set up Adam-trainable parameters.
		self.p      = {p.name:p for p in self.U + self.V + self.b + self.c}
		self.m1     = zerosLikeSVs(self.p, lambda x:x+"_m1")
		self.m2     = zerosLikeSVs(self.p, lambda x:x+"_m2")
		self.t      = T.shared(np.full((), 0,    dtype="uint64"))
	def consUs(self, isOrtho=False):
		U = []
		for i in xrange(1, len(self.config)):
			# Going up, "old" is the lower layer.
			oldN    = self.config[i-1]
			n       = self.config[i]
			U += [T.shared(initWeights(n, oldN, isOrtho), "U"+str(i))]
		return U
	def consVs(self, isOrtho=False):
		V = []
		for i in xrange(1, len(self.config)):
			# Going down, "old" is the upper layer.
			oldN    = self.config[i]
			n       = self.config[i-1]
			V      += [T.shared(initWeights(n, oldN, isOrtho), "V"+str(i-1))]
		return V
	def consbs(self):
		b = []
		for i in xrange(1, len(self.config)):
			n       = self.config[i]
			b      += [T.shared(initBiases(n), "b"+str(i))]
		return b
	def conscs(self):
		c = []
		for i in xrange(0, len(self.config)-1):
			n       = self.config[i]
			c      += [T.shared(initBiases(n), "c"+str(i))]
		return c
	def f(self, i, h):
		"""Compute forward contribution into i'th layer from given input, which should
		represent i-1'th layer."""
		
		if(isinstance(h, TT.sharedvar.TensorSharedVariable) or
		   isinstance(h, TT.TensorVariable)):
			return self.b[i-1] + TT.dot(self.rho(h), self.U[i-1].T)
		else:
			return self.b[i-1].get_value() + np.dot(self.rho(h), self.U[i-1].get_value().T)
	def g(self, i, h):
		"""Compute backwards contribution into i'th layer from given input, which should
		represent i+1'th layer."""
		
		if(isinstance(h, TT.sharedvar.TensorSharedVariable) or
		   isinstance(h, TT.TensorVariable)):
			return self.c[i] + TT.dot(self.rho(h), self.V[i].T)
		else:
			return self.c[i].get_value() + np.dot(self.rho(h), self.V[i].get_value().T)
	def consTFs(self):
		"""Create reconstruction loss, training and step functions"""
		
		# Theano functions lists
		self.reconstructLossTFs = []
		self.trainTFs           = []
		self.stepTF             = None
		
		# Layers list
		self.h   = [TT.matrix()] # Input (h0), formatted BatchSize x InputVectorLength
		
		#
		# Create K reconstruction and training functions, one for each stacked AE and
		# one "global" fine-tuning version.
		#
		
		noise = TT.matrix()
		for i in xrange(len(self.config)-1):
			# Compute reconstruction
			hOld    = self.h[i]
			self.h += [self.f(i+1, hOld)]
			hNew    = self.g(i, self.f(i+1, hOld+noise))
			
			# Compute squared-error reconstruction error between hOld and its
			# reconstruction hNew
			sqdiff     = (hOld-hNew)**2
			loss       = TT.sum(TT.mean(sqdiff, axis=0))
			l2Regul    = self.hL2P*reduce(lambda a,b:a+b, map(lambda x: TT.sum(x**2), self.U+self.V))
			actRegul   = 0 #(TT.mean(TT.sqrt(TT.sum((hNew)**2, axis=1)), axis=0)-1)**2
			regul      = l2Regul + actRegul
			
			# Add to list of reconstruction functions
			self.reconstructLossTFs += [T.function([self.h[0], noise], [TT.sqrt(loss)])]
			self.trainTFs           += [T.function([self.h[0], noise], [TT.sqrt(loss)],
		                                          updates=self.consUPDs(loss+regul, i))]
		
		
		#
		# Create step function to report the magnitude of the step size at each
		# relaxation iteration.
		#
		
		eps        = TT.scalar()
		self.oldH  = []
		for i in xrange(0, len(self.config)):
			self.oldH += [TT.matrix()]
		self.newH = self.oldH[:]
		
		#
		# Noise inputs
		#
		
		relaxNoise = [TT.matrix() for i in xrange(1, len(self.config))]
		
		
		# Resample odd layers
		for i in xrange(1, len(self.config), 2):
			if i == len(self.config)-1:
				# Last layer gets special treatment
				self.newH[i] = (1-eps)*self.newH[i] - eps*(self.f(i, self.newH[i-1])) + relaxNoise[i-1]
			else:
				hTerm  = self.newH[i]
				gTerm  = self.g(i, self.newH[i+1])
				gfTerm = self.g(i, self.oldH[i+1])
				
				self.newH[i] = (1-eps)*self.newH[i] - eps*0.5*(self.f(i, self.newH[i-1])+
				                                               gTerm) + relaxNoise[i-1] #hTerm+gTerm-gfTerm)
		# Resample even layers
		for i in xrange(2, len(self.config), 2):
			if i == len(self.config)-1:
				# Last layer gets special treatment
				self.newH[i] = (1-eps)*self.newH[i] - eps*(self.f(i, self.newH[i-1])) + relaxNoise[i-1]
			else:
				hTerm  = self.newH[i]
				gTerm  = self.g(i, self.newH[i+1])
				gfTerm = self.g(i, self.oldH[i+1])
				
				self.newH[i] = (1-eps)*self.newH[i] - eps*0.5*(self.f(i, self.newH[i-1])+
				                                               gTerm) + relaxNoise[i-1] #hTerm+gTerm-gfTerm)
		
		# Compute squared-error reconstruction error between each h and its hp
		self.oldNewSqdiffs    = map(lambda x: TT.mean(TT.sum((x[0]-x[1])**2, axis=1), axis=0), zip(self.oldH, self.newH))
		self.oldMag           = TT.sqrt(reduce(lambda a,b:a+b, map(lambda x: TT.mean(TT.sum(x**2, axis=1), axis=0), self.oldH)))
		self.oldNewLoss       = TT.sqrt(reduce(lambda a,b:a+b, self.oldNewSqdiffs))
		self.stepTF           = T.function([eps] + self.oldH + relaxNoise, self.newH + [self.oldNewLoss, self.oldMag])
	def consUPDs(self, loss, i):
		if i<len(self.config):
			U    = self.U[i]
			V    = self.V[i]
			b    = self.b[i]
			c    = self.c[i]
			
			U_m1 = self.m1[U.name]
			V_m1 = self.m1[V.name]
			b_m1 = self.m1[b.name]
			c_m1 = self.m1[c.name]
			
			U_m2 = self.m2[U.name]
			V_m2 = self.m2[V.name]
			b_m2 = self.m2[b.name]
			c_m2 = self.m2[c.name]
			
			U_g  = T.grad(loss, U)
			V_g  = T.grad(loss, V)
			b_g  = T.grad(loss, b)
			c_g  = T.grad(loss, c)
			
			return Adam([U,  V,  b,  c  ], [U_m1,V_m1,b_m1,c_m1], [U_m2,V_m2,b_m2,c_m2],
			            [U_g,V_g,b_g,c_g], self.t, self.hLrn, self.hMom)
		else:
			p  = self.p.values()
			m1 = [self.m1[x.name] for x in p]
			m2 = [self.m1[x.name] for x in p]
			g  = [T.grad(loss, p) for x in p]
			
			return Adam(p, m1, m2, g, self.t, self.hLrn, self.hMom)
	def rho(self, x):
		if(isinstance(x, TT.sharedvar.TensorSharedVariable) or
		   isinstance(x, TT.TensorVariable)):
			return TT.clip(x, 0.0, 1.0)                          # Hard Sigmoid
		else:
			return np.minimum(np.maximum(x,0.0),1.0)
	def fastInit(self, h):
		hs = [h.astype(TC.floatX)]
		
		for i in xrange(1, len(self.config)):
			hs += [self.f(i, hs[i-1])]
		
		return hs
	def naiveInit(self, h):
		hs = [h.astype(TC.floatX)]
		
		for i in xrange(1, len(self.config)):
			hs += [np.random.normal(0,1,size=(h.shape[0], self.config[i])).astype(TC.floatX)]
		
		return hs
	def resetAdam(self):
		self.t   .set_value(np.full((), 0,              dtype="uint64"))
		self.hLrn.set_value(np.full((), self.hLrnBase,  dtype=TC.floatX))
		for m in self.m1.itervalues():
			m.set_value(np.zeros_like(m.get_value()))
		for m in self.m2.itervalues():
			m.set_value(np.zeros_like(m.get_value()))
	def run(self, trainX, trainY, savePath, naive=False):
		# Savefile
		f = H.File(savePath)
		shouldReload = "params/" in f
		
		
		#
		# Should we train?
		#
		
		maxE = 30
		bS   = 50
		numB = len(trainX)/bS
		if not naive and not shouldReload:
			# Train for each layer:
			for l in xrange(len(self.config)-1):
				self.resetAdam()
				
				# For each batch:
				for e in xrange(maxE):
					# Set learning rate parameter:
					self.hLrn.set_value(np.full((), self.hLrn.get_value()*0.97, dtype=TC.floatX))
					totalLoss = 0.0
					
					# Print epoch header
					sys.stdout.write("Layer {:2d}, Epoch {:2d}\n".format(l, e))
					sys.stdout.flush()
					
					# Train on each batch:
					for b in xrange(len(trainX)/bS):
						batch = trainX[b*bS:(b+1)*bS]
						#pdb.set_trace()
						noise = np.random.normal(0, 0.1, size=(bS, self.config[l])).astype(TC.floatX)
						loss  = self.trainTFs[l](batch, noise)[0][()]
						totalLoss += loss
						sys.stdout.write("Reconstruction Loss: {:20.17f}\r".format(totalLoss/(b+1)))
						sys.stdout.flush()
					
					# Print epoch trailer
					sys.stdout.write("\n")
					sys.stdout.flush()
		
		if not shouldReload:
			# Save weights
			for p in self.p.itervalues():
				try:
					f.create_dataset("params/"+p.name, data=p.get_value())
				except:
					print "Error creating dataset ", p.name
					quit()
		else:
			# Reload weights
			for p in self.p.itervalues():
				p.set_value(f["params/"+p.name][...])
		
		
		# Now, use them.
		TEST_STEPS = 20
		
		if "log/stepsize" in f:
			logStepSize      = f["log/stepsize"]
		else:
			logStepSize      = f.require_dataset("log/stepsize",      shape=(TEST_STEPS,), maxshape=(None,), dtype=TC.floatX)
		logStepSize.resize(TEST_STEPS, 0)
		
		if "log/activationMag" in f:
			logActivationMag = f["log/activationMag"]
		else:
			logActivationMag = f.require_dataset("log/activationMag", shape=(TEST_STEPS,), maxshape=(None,), dtype=TC.floatX)
		logActivationMag.resize(TEST_STEPS, 0)
		
		
		# Initialization scheme?
		if naive:
			h = self.naiveInit(trainX[0:50])
		else:
			h = self.fastInit(trainX[0:50])
		
		# Create a history.
		history = [np.empty((0,bS,l), dtype=TC.floatX) for l in self.config]
		for i in xrange(len(self.config)):
			history[i] = np.concatenate([history[i], h[i].reshape((1,bS,self.config[i]))])
		
		# Relax.
		for i in xrange(TEST_STEPS):
			ret = self.stepTF(0.5*(1.0**i), *(h+self.genStepNoise(0.1, bS)))
			h = ret[:-2]
			for i in xrange(len(self.config)):
				history[i] = np.concatenate([history[i], h[i].reshape((1,bS,self.config[i]))])
			logStepSize[i]      = ret[-2][()]
			logActivationMag[i] = ret[-1][()]
			sys.stdout.write("Step Size: {:20.17f}   Activation Magnitude: {:20.17f}\n".format(logStepSize[i], logActivationMag[i]))
			sys.stdout.flush()
		
		# Analyze history
		# Average last 10 from history in each layer
		last10Avg = []
		for i in xrange(len(self.config)):
			last10Avg += [np.mean(history[i][-10:], axis=0, keepdims=True)]
		
		# Compute L2 distance to this average vector.
		distance2Avg = np.zeros((len(history[0]),), dtype=TC.floatX)
		for i in xrange(len(self.config)):
			distance2Avg += np.mean(np.sum((history[i]-last10Avg[i])**2, axis=2), axis=1)
		
		# Print these distances
		print distance2Avg
		
	def genStepNoise(self, sigma, bS):
		noise = []
		
		for i in xrange(1, len(self.config)):
			N = np.random.normal(0, sigma, size=(bS, self.config[i])).astype(TC.floatX)
			noise += [N]
		
		return noise


###############################################################################
# Implementations of the script's "verbs".
#

def verb_help(argv=None):
	print """Run this script with a verb argument, like "train"."""
def verb_train(argv=None):
	"""Train."""
	
	grn = GRN([3072, 256, 256, 256, 256, 256])
	#grn.run(*loadMNIST(), savePath=argv[2], naive=False)
	grn.run(*loadCIFAR10(), savePath=argv[2], naive=True)
def verb_screw(argv=None):
	"""Screw around."""
	
	pass
def verb_dumpimage(arg=None):
	import pylab
	import matplotlib.pyplot as plt
	
	f1NTNO = H.File("save-nontrained-nonortho-1layer.hdf5", "r")
	f6NTNO = H.File("save-nontrained-nonortho-6layer.hdf5", "r")
	f1TO   = H.File("save-trained-ortho-1layer.hdf5", "r")
	f6TO   = H.File("save-trained-ortho-6layer.hdf5", "r")
	
	fig   = plt.figure(figsize=(10,5))
	ax    = fig.add_subplot(1,1,1)
	
	ax.set_title("Relaxation Step Size")
	ax.set_xlabel("Step #")
	ax.set_ylabel("L2 Norm of Step Size")
	ax.tick_params(which="both", top="off", right="off")
	ax.set_xlim([0, 100])
	
	#print f1TO["/log/stepsize"][:]
	#pdb.set_trace()
	
	#ax.plot(xrange(1,101), f1NTNO["/log/stepsize"][:], linewidth=2, color="red",   label="Random (1-layer)")
	#ax.plot(xrange(1,101), f1TO  ["/log/stepsize"][:], linewidth=2, color="blue",  label="Auto-Encoder (1-layer)")
	ax.plot(xrange(1,101), f6NTNO["/log/stepsize"][:], linewidth=2, color="green", label="Random (6-layer)")
	ax.plot(xrange(1,101), f6TO  ["/log/stepsize"][:], linewidth=2, color="black", label="Auto-Encoder (6-layer)")
	
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
