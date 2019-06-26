#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    data.append( (x,y) )
  return (data, varnames)


# Sigmoid function - map to value between 0 and 1
def sigmoid(z):
  try:
    return 1.0 / (1.0 + exp(-z))
  except OverflowError:
    return 0


# Dot product
def dot(v1, v2):
  return sum([i*j for (i, j) in zip(v1, v2)])


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):

  # Initialize weights and bias
  numvars = len(data[0][0])
  w = [0.0] * numvars
  b = 0.0

  # Get x and y values from data
  X = []
  Y = []
  for (x, y) in data:
    X.append(x)
    Y.append(y)
  X_col = list(zip(*X))

  for i in range(MAX_ITERS):
    # Calculate predicted values for each sample in data
    predicted = []
    for x in X:
      prob = dot(w,x) + b
      predicted.append(prob)

    # Calculate the difference between predicted - actual
    residuals = [-y * sigmoid(-y * yhat) for y, yhat in zip(Y, predicted)]

    # Calculate gradients for weights (dws) and bias (db)
    dws = []
    for j in range(len(X_col)):
      dw = dot(X_col[j], residuals) + l2_reg_weight * w[j]    # regularized weight
      dws.append(dw)
    db = sum(residuals)     # bias

    # Check if converged and return
    if sqrt(sum([k**2 for k in dws] + [db**2])) < 0.0001:
      return (w,b)

    # Update weights and bias
    for d in range(len(X_col)):
      w[d] = w[d] - (eta * dws[d])
    b = b - (eta * db)

  # Return if not converged after maxiters
  return (w,b)


# Predict the probability of the positive label (y=+1) given the attributes, x.
def predict_lr(model, x):
  (w,b) = model
  return sigmoid(dot(w, x) + b)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 5):
    print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  eta = float(argv[2])
  lam = float(argv[3])
  modelfile = argv[4]

  # Train model
  (w,b) = train_lr(train, eta, lam)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    prob = predict_lr( (w,b), x )
    #print(prob)
    if (prob - 0.5) * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
