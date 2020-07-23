import numpy as np
from hmm.continuous.EMAGMHMM import EMAGMHMM
from hmm.continuous.GMHMM import GMHMM
import sys
np.set_printoptions(threshold=sys.maxsize)

def generate(M, K, D, tmat, means, covars, weights, pi, T=10):
  model = GMHMM(M, K, D, tmat, means, covars, weights, pi, init_type='user')
  obs = np.zeros((T, 2))
  s = np.random.choice(M, p=pi) # pretend we are starting from pi[0]
  r = np.random.choice(K, p=weights[s])
  obs[0] = np.random.multivariate_normal(means[s][r], covars[s][r])
  for t in range(1, T):
    s = np.random.choice(M, p=tmat[s]) # choose state
    r = np.random.choice(K, p=weights[s]) # choose mixture
    obs[t] = np.random.multivariate_normal(means[s][r], covars[s][r])

  #print(obs)
  prob = model.forwardbackward(obs)
  print("Real LL= ", prob)

  return obs

