#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

# import pybn library
from pybn import *

  # Define a main() function.
def main():

  # Create a Network
  net = Network('CarProblem')

  # Create Node 'Fuel'
  Fu = Node('Fu')

  # Setting number (and name) of outcomes
  Fu.addOutcome('yes')
  Fu.addOutcome('no')

  # Create node 'Clean Spark Plugs'
  SP = Node('SP')

  # Setting number (and name) of outcomes
  SP.addOutcomes(['yes','no'])

  # Create node 'Fuel Meter Standing'
  FM = Node('FM')

  # Setting number (and name) of outcomes
  FM.addOutcomes(['full','half','empty'])

  # Create node 'Start'
  St = Node('St')

  # Setting number (and name) of outcomes
  St.addOutcomes(['yes','no'])

  # Add arc from 'Fuel' to 'Fuel Meter Standing'
  arc_Fu_FM = Arc(Fu,FM)

  # Add arc from 'Fuel' to 'Start'
  arc_Fu_St = Arc(Fu,St)

  # Add arc from 'Clean Spark Plugs' to 'Start'
  arc_SP_St = Arc(SP,St)


  # Shows the size of the probability matrix 'Start'
  # print FM.getTableSize()

  # Conditional distribution for node 'Fuel'
  Fu.setProbabilities([0.98,0.02])

  # Conditional distribution for node 'Clean Spark Plugs'
  SP.setProbabilities([0.96,0.04])

  # Conditional distribution for node 'Fuel Meter Standing'
  FM.setProbabilities([0.39, 0.60, 0.01, 0.001, 0.001, 0.998])

  # Conditional distribution for node 'Start'
  St.setProbabilities([0.99, 0.01, 0.01, 0.99, 0.0, 1.0, 0.0, 1.0])


  # Changing the nodes spacial and visual attributes:
  Fu.setNodePosition(100,10)

  SP.setNodePosition(300,10)

  FM.setNodePosition(0,150)
  FM.setInteriorColor('cc99ff')

  St.setNodePosition(200,150)
  St.setInteriorColor('ff0000')

  # Add notes to network
  net.addNodes([Fu,SP,FM,St])

  # Write file
  net.writeFile('CarProblem.xdsl')

  # Set evidence
  net.setEvidence('FM',2)
  net.setEvidence('St',2)

  # Compute the beliefs for the network
  net.computeBeliefs()

  # Print the results for each node
  print('Fu', Fu.getBeliefs())
  print('SP', SP.getBeliefs())
  print('St', St.getBeliefs())
  print('FM', FM.getBeliefs())

  # This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()

