import numpy
import numpy as np
from genmodel import generate
from hmm.continuous.GMHMM import GMHMM
np.seterr(all='raise')
np.random.seed(1000)

M = 5
K = 4
D = 2

atmp = np.random.random_sample((M, M))
row_sums = atmp.sum(axis=1)
tmat = np.array(atmp / row_sums[:, np.newaxis], dtype=np.double)

wtmp = np.random.random_sample((M, K))
row_sums = wtmp.sum(axis=1)
w = np.array(wtmp / row_sums[:, np.newaxis], dtype=np.double)

means = np.array((0.6 * np.random.random_sample((M, K, D)) - 0.3), dtype=np.double)
covars = np.zeros( (M,K,D,D) )

for i in range(M):
    for j in range(K):
      covars[i][j] = np.matrix([[1,0],[0,1]])

pitmp = np.random.random_sample((M))
pi = np.array(pitmp / sum(pitmp), dtype=np.double)
obs = generate(M, K, D, tmat, means, covars, w, pi, 100)

#print(obs)

xpi = np.ones( (M)) *(1.0/M)
xA = np.ones( (M,M))*(1.0/M)
xw = np.ones( (M,K))*(1.0/K)
xmeans = np.array((1 * np.random.random_sample((M, K, D)) - 0.5), dtype=np.double)

#hmm = GMHMM(M, K, D, xA, xmeans, covars, xw, xpi, init_type='user', verbose=True)
hmm = GMHMM(M, K, D, verbose=True)
#hmm.covars[:] +=  1e-7 * np.eye(D)

covars = np.zeros( (M,K) )

covars = [[ numpy.matrix(numpy.ones((D,D))) for j in range(K)] for i in range(M)]
for i in range(M):
    for j in range(K):
      covars[i][j] = np.matrix([[1.000001,1],[1,1.0000001]])
hmm.covars = covars
print(hmm.covars)

hmm.train(obs, iterations=10)
