'''
Created on Jul 18, 2017

@author: damien.thioulouse
'''

from scipy import r_
import numpy as np

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.rprop import RPropMinusTrainer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import one_to_n
import time
import multiprocessing
from pybrain.tools.datasettools import DataSetNormalizer
from pybrain.tools.customxml.networkwriter import NetworkWriter

def unwrap_self_f(arg):
    return NFQ1.learnSequence(*arg)

class NFQ1(ValueBasedLearner):
    """ Neuro-fitted Q-learning"""
    def __init__(self, maxEpochs=20):
        ValueBasedLearner.__init__(self)
        self.gamma = 0.8
        self.maxEpochs = maxEpochs
        
    def learnSequence(self,seq):
        #print seq
        lastexperience = None
        o = 0
        #print len(seq) 
        out = [None] * (len(seq)-1) 
        for state, action, reward in seq:
            if not lastexperience:## how to handle sequences of 1 element? 
                # delay each experience in sequence by one
                lastexperience = (state, action, reward)
                continue
            
            # use experience from last timestep to do Q update
            (state_, action_, reward_) = lastexperience
            #Q = self.module.getValue(state_, action_[0])
            
            inp = r_[state_, one_to_n(action_[0], self.module.numActions)] #create numpy array
            #tgt = Q + 0.5*(reward_ + self.gamma * max(self.module.getActionValues(state)) - Q)
            tgt = reward_ + self.gamma * max(self.module.getActionValues(state))
            
            # update last experience with current one
            lastexperience = (state, action, reward)
            out[o] = (inp,tgt)
            o += 1 
        return out
    
    def buildSupervised(self):
        # convert reinforcement dataset to NFQ supervised dataset
        supervised = SupervisedDataSet(self.module.network.indim, 1)
        #print "multiprocess"
        # self dataset is in the form: (array([ 0.54906412,  4.53764151,  0.3166573 , -1.08502779]), array([ 1.]), array([ 0.]))
        print "Build Dataset on ", multiprocessing.cpu_count(), " CPUs..."
        t0 = time.time()
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        result = p.map(unwrap_self_f,zip([self]*(self.dataset.getNumSequences()), self.dataset), chunksize=100)
        p.close()
        p.join()
        #print "result"#, result
        #print np.shape(result)
        #print type(result)
        #print result[10]
        
        #print np.shape([item for sublist in result for item in sublist])
        #result1 = [x for x in result if x is not None]
        #total = [1 for x in result if x == None]
        #print "sum:", sum(total)
        #print result1[10]
        #print type(result1)
        #print np.shape(result1)
        result2 = [item for sublist in result for item in sublist]
        #print result1
        #print result2[10]
        t1 = time.time()
        print "Multiprocessing done"
        print "Time preparation: ", t1-t0
        #print "original shape: ", Rdata["state"].shape[0]
        #print "end shape: ", np.shape(result2)[0]
        i = 0 
        for inp, tgt in result2:
#            i += 1
#             if i%1000 == 0:
#                 print i     
            supervised.addSample(inp, tgt)
            #print "added"
            #print tgt #supervised.addSample(inp, tgt)
        if self.pathOutputDataset != None:
            print "writing to: ", self.pathOutputDataset
            supervised.saveToFile(self.pathOutputDataset)
        
        
    def learn(self):
        
        print "reading supervised dataset from ", self.pathInputDataset
        supervised = SupervisedDataSet.loadFromFile(self.pathInputDataset)  
         
        if self.module.normalizerState == None:
            normSup = DataSetNormalizer()
            normSup.calculate(supervised["input"][:,:self.module.indim])
            self.module.normalizerState = normSup
            normSup.save(self.pathNormalizerSup)
            print "Saved state normalizer to ", self.pathNormalizerSup
        
        normalizedStates = self.module.normalizerState.normalize(supervised["input"][:,:self.module.indim])
        both = np.concatenate((normalizedStates,supervised["input"][:,self.module.indim:]),axis=1)
        supervised.setField("input", both)
        
        normTar = DataSetNormalizer()
        normTar.calculate(supervised["target"])
        self.module.normalizerTarget = normTar
        
        print "Saving output normalizer to ", self.pathNormalizerTar
        normTar.save(self.pathNormalizerTar)
        normalizedTarget = self.module.normalizerTarget.normalize(supervised["target"])
        supervised.setField("target", normalizedTarget)
        
        print "Normalization of input and target completed"
        
        print supervised
        
        # train module with backprop/rprop on dataset
        print "Training Starting..."
        #print supervised
        t2 = time.time()
        trainer = RPropMinusTrainer(self.module.network, dataset=supervised, batchlearning=True, verbose=True, weightdecay=0.0001)
        trainer.trainUntilConvergence(maxEpochs=60,validationProportion=0.40)
        
        print "Training Completed"
        
        if self.pathNnetWrite != None:
            print "writing to: ", self.pathNnetWrite
            NetworkWriter.writeToFile(self.module.network, self.pathNnetWrite)
        
        t3 = time.time()
        
        print "Total time training: ", t3-t2
        #return supervised
        
        # alternative: backprop, was not as stable as rprop
        # trainer = BackpropTrainer(self.module.network, dataset=supervised, learningrate=0.005, batchlearning=True, verbose=True)
        # trainer.trainUntilConvergence(maxEpochs=self.maxEpochs)