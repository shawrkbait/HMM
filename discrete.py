'''
Created on Nov 13, 2012

@author: GuyZ
'''

from hmm.continuous.GMHMM import GMHMM
from hmm.discrete.DiscreteHMM import DiscreteHMM
import numpy

    
def test_discrete():

    ob5 = (3,1,2,1,0,1,2,3,1,2,0,0,0,1,1,2,1,3,0)
    print("Doing Baum-welch")
    
    atmp = numpy.random.random_sample((4, 4))
    row_sums = atmp.sum(axis=1)
    a = atmp / row_sums[:, numpy.newaxis]    

    btmp = numpy.random.random_sample((4, 4))
    row_sums = btmp.sum(axis=1)
    b = btmp / row_sums[:, numpy.newaxis]
    
    pitmp = numpy.random.random_sample((4))
    pi = pitmp / sum(pitmp)
    
    hmm2 = DiscreteHMM(4,4,a,b,pi,init_type='user',precision=numpy.longdouble,verbose=True)
    hmm2.train(numpy.array(ob5*10),100)
    print("Pi",hmm2.pi)
    print("A",hmm2.A)
    print("B", hmm2.B)
    print(hmm2.decode(numpy.array(ob5*10)))
    
    
test_discrete()
