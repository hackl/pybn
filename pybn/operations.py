#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib

class Factor(object):
  """Factor

  Factors are used to represent the CPDs in the Bayesian network. As such, the
  core functionality, which are implemented, are the factor product, factor
  marginalization and factor reduction operations.

  :Attributes:
    - name (str): Name of the factor
    - var (list): List of variables in the factor, e.g. [1 2 3]
    - card (list): List of cardinalities corresponding to .var, e.g. [2 2 2]
    - val (list): Value table of size prod(card)
  """

  def __init__(self):
    self.name = None
    self.var = None
    self.card = None
    self.val = None

  def getVar(self):
    return self.var

  def getCard(self):
    return self.card

  def getVal(self):
    return self.val

  def setVar(self,var):
    self.var = np.array(var)

  def setCard(self,card):
    self.card = np.array(card)

  def setVal(self,val):
    self.val = np.array(card)

  def input(self, var,card,val):
    self.var = np.array(var)
    self.card = np.array(card)
    self.val = np.array(val)

  def __str__(self):
    return self.name


def FactorProduct(A,B):
  """Factor Product Computes the product of two factors.

  ``C.FactorProduct(A,B)`` computes the product between two factors, A and B,
  where each factor is defined over a set of variables with given dimension.
  The factor data structure has the following fields:

    - .var    Vector of variables in the factor, e.g. [1 2 3]
    - .card   Vector of cardinalities corresponding to .var, e.g. [2 2 2]
    - .val    Value table of size prod(.card)

  :Args:
    - A (Factor): Factor A
    - B (Factor): Factor B

  :Returns:
    - C (Factor): Return factor C
  """
  C = Factor()
  # Check for empty factors
  if A.var == np.array([]):
    C = B
    return C
  if B.var == np.array([]):
    C = A
    return C
  else:
    # Check that variables in both A and B have the same cardinality
    dummy = np.intersect1d(A.var, B.var)
    if dummy == np.array([]):
      print 'Dimensionality mismatch in factors'

    # Set the variables of C
    C.var = np.union1d(A.var, B.var)

    # Construct the mapping between variables in A and B and variables in C.
    # In the code below, we have that
    #    mapA(i) = j, if and only if, A.var(i) == C.var(j)
    # and similarly 
    #    mapB(i) = j, if and only if, B.var(i) == C.var(j)
    # For example, if A.var = [3 1 4], B.var = [4 5], and C.var = [1 3 4 5],
    # then, mapA = [2 1 3] and mapB = [3 4]; mapA(1) = 2 because A.var(1) = 3
    # and C.var(2) = 3, so A.var(1) == C.var(2).

    dummy, ixA = ismember(A.var,C.var)
    mapA = (A.var[dummy])
    dummy, ixB = ismember(B.var,C.var)
    mapB = (B.var[dummy])

    # Set the cardinality of variables in C
    C.card = np.zeros(len(C.var))
    C.card[ixA] = A.card
    C.card[ixB] = B.card

    # Initialize the factor values of C:
    #   prod(C.card) is the number of entries in C
    C.val = np.zeros(np.prod(C.card))

    # Compute some helper indices
    assignment = IndexToAssignment(np.arange(np.prod(C.card)),C.card)
    indxA =  AssignmentToIndex(assignment[:,ixA], A.card)
    indxB =  AssignmentToIndex(assignment[:,ixB], B.card)

    # Correctly populate the factor values of C
    for i in range(int(np.prod(C.card))):
      C.val[i] = A.val[indxA[i]]*B.val[indxB[i]]

    return C


def FactorMarginalization(A, V):
  """Factor Marginalization Sums given variables out of a factor.

  ``B = FactorMarginalization(A,V)`` computes the factor with the variables
  in V summed out. The factor data structure has the following fields:

    - .var    Vector of variables in the factor, e.g. [1 2 3]
    - .card   Vector of cardinalities corresponding to .var, e.g. [2 2 2]
    - .val    Value table of size prod(.card)

  The resultant factor should have at least one variable remaining or this
  function will throw an error.

  :Args:
    - A (Factor): Factor A
    - V (list): variables summed out

  :Returns:
    - B (Factor): Return factor B
  """

  B = Factor()

  # Check for empty factors
  if A.var == np.array([]) and V == np.array([]):
    B = A
    return B
  else:

    # Construct the output factor over A.var \ V
    # (the variables in A.var that are not in V)
    # and mapping between variables in A and B
    B.var, mapB = setdiff(A.var,V)

    # Check for empty resultant factor
    if B.var == np.array([]):
      print 'Error: Resultant factor has empty scope'
    else:

      # Initialize B.card and B.val
      B.card = A.card[mapB]
      B.val = np.zeros(np.prod(B.card))

      # Compute some helper indices
      assignment = IndexToAssignment(np.arange(np.prod(A.card)),A.card)
      indxB =  AssignmentToIndex(assignment[:,mapB], B.card)

      # Correctly populate the factor values of B
      for i in range(int(len(indxB))):
        B.val[indxB[i]] += A.val[i]

      return B


def ObserveEvidence(F, E):
  """Observe Evidence Modify a vector of factors given some evidence.

  ``F = ObserveEvidence(F, E)`` sets all entries in the vector of factors, F,
  that are not consistent with the evidence, E, to zero. F is a vector of
  factors, each a data structure with the following fields:

    - .var    Vector of variables in the factor, e.g. [1 2 3]
    - .card   Vector of cardinalities corresponding to .var, e.g. [2 2 2]
    - .val    Value table of size prod(.card)

  :Args:
    - F (Factor): Factor F
    - E (tuple):   E is an N-by-2 matrix, where each row consists of a variable/value pair.\n 
  Variables are in the first column and values are in the second column.

  :Returns:
    - F (Factor): Return factor F
  """
  # Iterate through all evidence
  for i in range(int(np.shape(E)[0])):
    v = E[i][0] # variable
    x = E[i][1] # value

    # Check validity of evidence
    # if x == 0:
    #   print 'Warning: Evidence not set for variable'

    for j in range(len(F)):
      # Does factor contain variable?
      indx = indices(F[j].var, lambda x: x == v)

      if indx != []:

        # Check validity of evidence
        if x > F[j].card[indx] or x < 0:
          print 'Error: Invalid evidence, X_'+str(v)+' = '+str(x)

        # Compute some helper indices
        assignment = IndexToAssignment(np.arange(np.prod(F[j].card)),F[j].card)
        idnxF = indices(F[j].var, lambda x: x == v)

        # Factor F(j) to account for observed evidence
        A = np.array([assignment[0]])
        for i in range(len(assignment)):
          if assignment[i][idnxF] != x:
            A = np.append(A,[assignment[i]],0)

        A = np.delete(A,0,0)
        F[j] = SetValueOfAssignment(F[j], A, 0,)

        # Check validity of evidence / resulting factor
        if F[j].val == np.array([]):
          print 'Warning: Factor '+str(j)+' makes variable assignment impossible'
  return F

def SetValueOfAssignment(F, A, v, VO=None):
  if VO == None:
    #print A
    #print F.card
    indx = AssignmentToIndex(A, F.card)
    # print indx.astype(int)
  else:
    map = [0]*len(F.var)
    for i in range(int(len(F.var))):
      map[i] = indices(VO, lambda x: x == F.var[i])
    indx = AssignmentToIndex(A[map], F.card)
  F.val[indx.astype(int)] = v

  return F


def ComputeJointDistribution(F):
  """Compute Joint Distribution computes the joint distribution defined by a set
  of given factors

  ``Joint = ComputeJointDistribution(F)`` computes the joint distribution
  defined by a set of given factors

  Joint is a factor that encapsulates the joint distribution given by F
  F is a vector of factors (struct array) containing the factors defining the
  distribution
  """

  Joint = Factor()
  # Check for empty factor list
  if F == []:
    print 'Error: empty factor list'
  elif len(F) == 1:
    print 'Error: only one factor'
  else:
    F.reverse()
    Joint = FactorProduct(F[0],F[1])
    if len(F) > 2:
      for i in range(2,len(F)):
        Joint = FactorProduct(Joint,F[i])
    return Joint


def ComputeMarginal(V, F, E):
  """Compute Marginal Computes the marginal over a set of given variables.

  ``M = ComputeMarginal(V, F, E)`` computes the marginal over variables V
  in the distribution induced by the set of factors F, given evidence E

  :Args:
    - V (list): V is a list containing the variables in the marginal e.g. [1 2 3] for X_1, X_2 and X_3.
    - F (list): F is a list of factors containing the factors defining the distribution
    - E (tuple): E is an N-by-2 matrix, each row being a variable/value pair.\n
    Variables are in the first column and values are in the second column.\n
    If there is no evidence, pass in the empty matrix [] for E.

  :Returns:
    - M (Factor): M is a factor containing the marginal over variables V
  """

  # Compute the joint distirbution
  J = ComputeJointDistribution(F)

  # Compute observed evidence
  E = ObserveEvidence([J], E)

  # Returns a renormalized factor
  R = RenormalizeFactor(E[0])

  # Store the results in a list
  M = []
  for i in range(len(V)):
    D = R
    for ii in range(len(R.var)):
      if R.var[ii] != V[i]:
        D = FactorMarginalization(D, [R.var[ii]])
    M.append(D)
  return M


def RenormalizeFactor(F):
  if F.val == np.array([]):
    print 'Error: Factor is empty'
  else:
    if np.sum(F.val) != 1:
      sum = np.sum(F.val)
      for i in range(len(F.val)):
        F.val[i] = F.val[i]*sum**(-1)
    return F


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def setdiff(a,b):
    tf = np.in1d(a,b)
    for i in range(len(tf)):
      if tf[i] == True:
        tf[i] = False
      else:
        tf[i] = True
    d = np.setdiff1d(a,b)
    index = np.array([np.where(a!=b)])[0][0]
    return d, index


def AssignmentToIndex(A, D):
  """AssignmentToIndex Convert assignment to index.

  ``I = AssignmentToIndex(A, D)`` converts an assignment, A, over variables
  with cardinality D to an index into the .val vector for a factor.
  If A is a matrix then the function converts each row of A to an index.

  :Args:
    - A (list): assignment A.
    - D (list): cardinality of the assignment

  :Returns:
    - I (list): Returns a list I with indices correlated to the assignment
  """
  if np.any(A.shape==1):
    I = np.dot(np.cumprod(np.append([1],D[:-1])),(np.reshape(A,-1,order='F')-1))-1
  else:
    I = np.sum(np.matlib.repmat(np.cumprod(np.append([1],D[:-1])),A.shape[0],1)*(A-1),1)
  return I


def ismember(a, b):
    tf = np.in1d(a,b) # for newer versions of numpy
    #tf = np.array([i in b for i in a])
    u = np.unique(a[tf])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
    return tf, index


def IndexToAssignment(I,D):
  """
  IndexToAssignment Convert index to variable assignment.

  ``A = IndexToAssignment(I, D)`` converts an index, I, into the .val vector
  into an assignment over variables with cardinality D. If I is a vector, 
  then the function produces a matrix of assignments, one assignment 
  per row.

  :Args:
    - I (list): List of indices
    - D (list): cardinality of the assignment

  :Returns:
    - A (list): Returns the assignment A related to the indices I.
  """
  I = I[np.newaxis].T
  A = np.mod(np.floor(np.matlib.repmat(I, 1,len(D)) / np.matlib.repmat(np.cumprod(np.append([1],D[:-1])), len(I),1)), np.matlib.repmat(D,len(I),1))+1
  return A


def intersect(a, b):
  return list(set(a) & set(b))



