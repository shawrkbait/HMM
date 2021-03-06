from hmm.continuous.EMAGMHMM import EMAGMHMM
import pandas as pd
import numpy as np

np.seterr(all='raise')
np.random.seed(1000)

spx = pd.read_csv('gspc.csv').sort_values(by='Date')
dji = pd.read_csv('dji.csv').sort_values(by='Date')

sdiff = 10 * spx['Adj Close'].pct_change()[1:]
ddiff = 10 * dji['Adj Close'].pct_change()[1:]

dates = spx['Date'][1:]


def testvals(start, end):
  oldval = 0
  a = []
  b = []
  for i in range(start,end):
    x = 0.2 * (np.random.rand()/100 - 0.005) + 0.7 * oldval + 0.1 * 0.00001
    a.append(10 *x)
    b.append(10 *(0.9 * x + 0.1 * (np.random.rand()/100 - .005)))
    oldval = x
  return np.column_stack([a,b])

def getEV(mu, R):
  print("State EVs")
  evs = []
  for m in range(len(mu)):
    ev = 0
    for n in range(len(mu[m])):
      ev += mu[m][n][0] * R[m][n]
    evs.append(ev)
  return evs

def test_rand():
    n = 5
    m = 4
    d = 2
    atmp = np.random.random_sample((n, n))
    row_sums = atmp.sum(axis=1)
    a = np.array(atmp / row_sums[:, np.newaxis], dtype=np.double)

    wtmp = np.random.random_sample((n, m))
    row_sums = wtmp.sum(axis=1)
    w = np.array(wtmp / row_sums[:, np.newaxis], dtype=np.double)

    means = np.array((6 * np.random.random_sample((n, m, d)) - 3), dtype=np.double)
    covars = np.zeros( (n,m,d,d) )

    for i in range(n):
        for j in range(m):
            for k in range(d):
                covars[i][j][k][k] = 1
    pitmp = np.random.random_sample((n))
    pi = np.array(pitmp / sum(pitmp), dtype=np.double)

    gmmhmm = EMAGMHMM(n,m,d,a,means,covars,w,pi,init_type='user',verbose=True, min_std=1e-8)

#    obs = testvals(0, 400)
    obs = np.column_stack([sdiff, ddiff])

    success = 0
    total = 0
    print("Doing Baum-welch")
    for i in range(40,400):
      seq = obs[i-40:i]
      print("Doing ", i-40)
      if i > 40:
        if nextev[int(viterbi[-1])] < 0 and seq[-1][0] < 0:
          success += 1
        elif nextev[int(viterbi[-1])] >= 0 and seq[-1][0] >= 0:
          success += 1
        total += 1
        print("%d/%d pred=%s actual=%s" % (success,total, nextev, np.array(seq[-1][0])))
      gmmhmm.train(seq,50, epsilon=1e-20)
      print(gmmhmm.means)
      print(gmmhmm.covars)
      viterbi = gmmhmm.decode(obs[i-40:i])
      print(viterbi)
      nextev = np.array(getEV(gmmhmm.means, gmmhmm.w))
      gmmhmm = EMAGMHMM(n,m,d,gmmhmm.A,gmmhmm.means,gmmhmm.covars,gmmhmm.w,pi,init_type='user',verbose=True, min_std=1e-8)

test_rand()
