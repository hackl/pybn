#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import sys
from operations import *
from operator import mul

class Network(object):
  """Bayesian Network

  :Attributes:
    - name (str): Name of the network
  """

  def __init__(self, name):
    self.name = repEmptySpace(name)
    self.nodes = []
    self.evidence = []
    self.marginal = None

  def __str__(self):
    return self.name

  def addNode(self, node):
    """Add one node to the network

    :Args:
      - node (Node): Node element
    """
    self.nodes.append(node)

  def addNodes(self,nodes):
    """Add a list of nodes to the network

    :Args:
      - nodes (list): A list of Node elements
    """
    for node in nodes:
      self.addNode(node)

  def setEvidence(self,name,value):
    """Set evidence for a Node element

    :Args:
      - name (str): Name of the Node
      - value (int): Number of stage which is observed. Starting with 0.
    """
    for i in range(len(self.nodes)):
      if name == str(self.nodes[i]):
        var = self.nodes[i].getIdNum()
        outcomes = self.nodes[i].getOutcomes()
        if type(value) is str:
          for j in range(len(outcomes)):
            if value == outcomes[j]:
              val = j+1
        elif type(value) is int:
          val = value
        self.evidence.append([var,val])

  def getEvidence(self):
    """Return information about the evidence

    :Returns:
      - evidence (list): Return a list with all observed nodes. First entry is the node number, second entry the stage which is observed.
    """
    if self.evidence == []:
      evidence = [[0,0]]
    else:
      evidence = self.evidence
    return evidence

  def reset(self):
    """Reset all values"""
    self.evidence = []
    self.marginal = None
    for i in range(len(self.nodes)):
      self.nodes[i].setBeliefs(self.nodes[i].getProbabilities())
      self.nodes[i].setCard([])
      var = np.append(self.nodes[i].getVar(),self.nodes[i].getArcConnectionId()).tolist()
      self.nodes[i].setVar(var)
      self.nodes[i].setVal(self.nodes[i].transformProbabilities())

  def computeBeliefs(self):
    """Compute beliefs of the network"""
    order = []
    factors = []
    marginal = []

    for i in range(len(self.nodes)):
      factor = Factor()
      order.append(self.nodes[i].getIdNum())
      self.nodes[i].setVal(self.nodes[i].transformProbabilities())
      var, card, val = self.nodes[i].getInput()
      factor.input(var,card,val)
      factors.append(factor)

    evidence = self.getEvidence()

    for factor in factors:
      subfactors = []
      suborder = factor.var
      for idNum in suborder:
        node = self.nodes[int(idNum)-1]
        subfactor = Factor()
        var, card, val = node.getInput()

        subfactor.input(var,card,val)
        subfactors.append(subfactor)

      if len(suborder) != 1:
        M = ComputeMarginal(suborder, subfactors, evidence)
        for idx,idNum in enumerate(suborder):
          node = self.nodes[int(idNum)-1]
          beliefs = M[idx].val
          card = M[idx].card
          var = M[idx].var
          node.setBeliefs(beliefs)
          node.setCard(card)
          node.setVar(var)
          node.setVal(beliefs)

  def getBeliefs(self,vars=None):
    """Returns all beliefs of the network

    :Returns:
      - beliefs: Return a list with all beliefs of the network.
    """
    beliefs = []
    if vars != None:
      for i in range(len(self.marginal)):
        for j in range(len(vars)):
          if str(self.marginal[i][0]) == vars[j]:
            beliefs.append(self.marginal[i][1])
    else:
      for i in range(len(self.marginal)):
        beliefs.append(self.marginal[i][1])
    return beliefs


  def writeFile(self,filename):
    """Write an output file

    :Args:
      - filename (str): Name of the outputfile, If no name is defined the name of the network will be used.

    :Returns:
      - outputfile: The output files is saved in the local folder.

    :Raises:
      - Error: No nodes connected to the network!
      - Error: Node 'xy' has no outcomes!
      - Error: Probabilities for 'xy' doesn't match!\n
        Len of probabilities should be a but is b
      - Error: Probabilities for 'xy' doesn't sum up to 1.0!

    """
    self.checkInput()
    if filename == None:
      filename = self.name+'.xdsl'

    f = open(filename,'w')
    f.writelines(self.writeHeader())
    for node in self.nodes:
      f.write(node.printNode())
    f.writelines(self.writeBody())
    for node in self.nodes:
      f.write(node.printExtension())
    f.writelines(self.writeFooter())
    f.close()

  def writeHeader(self):
    header = ['<?xml version="1.0" encoding="ISO-8859-1"?>\n',
              '<smile version="1.0" id="'+self.name+'" numsamples="1000" discsamples="10000">\n',
              '\t<nodes>\n']
    return header

  def writeBody(self):
    body = ['\t</nodes>\n','\t<extensions>\n','\t\t<genie version="1.0" app="py2GeNIe 2013" name="'+self.name+'" faultnameformat="nodestate">\n']
    return body

  def writeFooter(self):
    footer = ['\t\t</genie>\n','\t</extensions>\n','</smile>']
    return footer

  def checkInput(self):
    if self.nodes == []:
      sys.exit("Error: No nodes connected to the network!")
    else:
      for node in self.nodes:
        if node.getOutcomes() == []:
          sys.exit("Error: Node '"+str(node)+"' has no outcomes!")
        m,n = node.getTableSize()
        nodeLen = m*n
        if len(node.getProbabilities()) != nodeLen:
          sys.exit("Error: Probabilities for '"+str(node)+"' doesn't match!\n       Len of probabilities should be "+str(nodeLen)+" but is "+str(len(node.getProbabilities())))
        n = n+0.0
        if str(sum(node.getProbabilities())) != str(n):
          print "Error: Probabilities for '"+str(node)+"' doesn't sum up to 1.0!"
          #sys.exit("Error: Probabilities for '"+str(node)+"' doesn't sum up to 1.0!")


class Node(object):
  """Node element for the Bayesian Network

  :Attributes:
    - name (str): Name of the node
  """
  nextIdNum = 1
  def __init__(self, name):
    self.name = repEmptySpace(name)
    self.caption = name
    self.idNum = Node.nextIdNum
    self.nodeId = 'Node_'+str(self.idNum)
    Node.nextIdNum += 1
    self.outcomes = []
    self.probabilities = []
    self.nextIdOut = 0
    self.arcConnection = []
    self.var = []
    self.card = []
    self.val = []
    self.var.append(self.idNum)
    self.beliefs = None

    self.interior_color = 'e5f6f7'
    self.outline_color = '0000bb'
    self.font_color = '000000'
    self.font_name = 'Arial'
    self.font_size = 8
    self.node_size = [0,0,125,65]
    self.node_position = [0,0,125,65]
    self.bar_active = True
    """View as Bar Chart

    :Default:
      - ``False``: i.e. GeNIe will show Icons instead Bar Charts by default
    """

  def __repr__(self):
    return self.name

  def getIdNum(self):
    return self.idNum

  def getName(self):
    return self.name

  def setCard(self,card):
    """Set cardinality of the factor

    :Args:
      - card (list): Cardinality of the factor
    """
    self.card = card

  def getCard(self):
    """Returns the cardinality of the factor

    :Returns:
      - card (list): Returns a list of cardinalities corresponding to variables of the network.
    """
    return self.card

  def setVar(self,var):
    """Set variables in the factor

    :Args:
      - var (list): List of variables (nodes) in the factor
    """
    self.var = var

  def getVar(self):
    """Returns the variables in the factor

    :Returns:
      - var (list): List of variables (nodes) in the factor
    """
    return self.var

  def setVal(self,val):
    """Set values of the node

    :Args:
      - val (list): List of values for the node
    """
    self.val = val

  def getVal(self):
    """Returns values of the node

    :Returns:
      - val (list): Return a list of values for the node
    """

    return self.val

  def setBeliefs(self,beliefs):
    """Set beliefs for the node

    :Args:
      - beliefs (list): List of beliefs for the node
    """
    self.beliefs = beliefs

  def getBeliefs(self):
    """Returns beliefs for the node

    :Args:
      - beliefs (list): Returns a list of beliefs for the node
    """
    return self.beliefs

  def addOutcome(self,name):
    """Add one outcome to the node

    :Args:
      - name (str): Name of the outcome
    """
    self.outcomes.append(repEmptySpace(name))

  def addOutcomes(self,names):
    """Add a list of outcomes to the node

    :Args:
      - names (list): A list of names for the outcomes
    """
    for name in names:
      self.outcomes.append(repEmptySpace(name))

  def getOutcomes(self):
    """Returns a list of outcomes

    :Returns:
      - outcomes (list): Returns a list of outcomes
    """
    return self.outcomes

  def setProbabilities(self,probabilities):
    """Set the probabilities for the node

    The order of these probabilities is given by considering the state of the
    first parent of the node as the most significant coordinate, then the
    second parent, then the third (and so on), and finally considering the
    coordinate of the node itself as the least significant one.

    :Args:
      - probabilities (list): A list of probabilities for the node
    """
    self.probabilities = probabilities
    self.val = self.transformProbabilities()
    self.beliefs = probabilities

  def transformProbabilities(self):
    """Transform the probabilities for the node

    :Returns:
      - probabilities (list): Returns a list of transformed probabilities
    """
    card = self.getCard()
    probabilities = self.probabilities

    if len(card) != 1:
      card.insert(len(card), card.pop(0))
      assignment = IndexToAssignment(np.arange(np.prod(card)),card)
      assignment = assignment.tolist()
      for i in range(len(card)-1,-1,-1):
        assignment.sort(key=lambda x: x[i])

      for i,item in enumerate(assignment):
        item.append(probabilities[i])

      assignment.sort(key=lambda x: x[len(card)-1])

      for i in range(len(card)-1):
        assignment.sort(key=lambda x: x[i])

      probabilities = []
      for item in assignment:
        probabilities.append(item[-1])

    return probabilities

  def getProbabilities(self):
    """Returns a list of probabilities

    :Returns:
      - probabilities (list): Returns a list of probabilities
    """
    return self.probabilities

  def getProbability(self,index):
    """Returns a single probabilitiy

    :Args:
      - index (int): Position in the list of probabilities

    :Returns:
      - probabilitiy (float): Returns a single probabilitiy out of the list of probabilities
    """
    return self.probabilities[index]

  def getArcConnection(self):
    return self.arcConnection

  def getArcConnectionId(self):
    ids = []
    for i in range(len(self.arcConnection)):
      ids.append(self.arcConnection[i][1])
    return ids

  def addArcConnection(self,name,id,size):
    self.arcConnection.append([name,id,size])
    self.var.append(id)

  def getTableSize(self):
    """Returns the size of the probability table

    :Retunrs:
     - m,n (tuple): m represents the rows (number of outcomes)\n
       n represents the columns (depending from the parents and their outcomes)
    """
    if self.arcConnection == []:
      size = (self.getSize(),1)
    else:
      n = 1
      for i in range(len(self.arcConnection)):
        n *= self.arcConnection[i][2]
      size = (self.getSize(),n)
    return size

  def getCard(self):
    """Returns the cardinality of the factor

    :Returns:
      - card (list): Returns a list of cardinalities corresponding to variables of the network.
    """
    if self.card == []:
      card = []
      if self.arcConnection == []:
        card.append(self.getSize())
      else:
        card.append(self.getSize())
        for i in range(len(self.arcConnection)):
          card.append(self.arcConnection[i][2])
    else:
      card = self.card.tolist()
    return card

  def getInput(self):
    """Returns the input values for the Bayesian network

    :Retunrs:
     - var,card,val (tuple): var variables in the factor\n
       card cardinalities corresponding to var\n
       val value of the node
    """
    card = self.getCard()
    return self.var, card, self.val

  def getTable(self):
    return self.tableSize

  def getSize(self):
    return len(self.outcomes)

  def printNode(self):
    # print node
    commentNode = '\t\t<!-- create node "'+self.caption+'" -->\n'
    initNode = '\t\t<cpt id="'+self.name+'" >\n'
    commentOutcomes = '\t\t\t<!-- setting names of outcomes -->\n'
    initOutcomes = ''
    for outcomes in self.outcomes:
      initOutcomes += '\t\t\t<state id="'+outcomes+'" />\n'

    if self.arcConnection != []:
      # print arc
      commentArc = '\t\t\t<!-- add arcs -->\n'
      initArc = '\t\t\t<parents>'
      for i in range(len(self.arcConnection)):
        initArc += self.arcConnection[i][0]+' '
      endArc =  '</parents>\n'
    else:
      commentArc = ''
      initArc = ''
      endArc = ''

    # print probabilities
    commentProbabilities = '\t\t\t<!-- setting probabilities -->\n'
    initProbabilities = '\t\t\t<probabilities>'
    for i,probability in enumerate(self.probabilities):
      initProbabilities += str(probability)+' '
    endProbabilities = '</probabilities>\n'
    endNode = '\t\t</cpt>\n'
    print_node =  commentNode+initNode+commentOutcomes+initOutcomes+commentArc+initArc+endArc+commentProbabilities+initProbabilities+endProbabilities+endNode
    return print_node

  def printExtension(self):
    initExtensions = '\t\t\t<node id="'+self.name+'">\n'
    initName = '\t\t\t\t<name>'+self.caption+'</name>\n'
    initIcolor = '\t\t\t\t<interior color="'+self.interior_color+'" />\n'
    initOcolor = '\t\t\t\t<outline color="'+self.outline_color+'" />\n'
    initFont = '\t\t\t\t<font color="'+self.font_color+'" name="'+self.font_name+'" size="'+str(self.font_size)+'" />\n'
    initPos = '\t\t\t\t<position>'+str(self.node_position[0])+' '+str(self.node_position[1])+' '+str(self.node_position[2])+' '+str(self.node_position[3])+'</position>\n'
    initBar = ''
    if self.bar_active == True:
      initBar = '\t\t\t\t<barchart active="true" width="'+str(self.node_size[2])+'" height="'+str(self.node_size[3])+'" />\n'
    endExtensions = '\t\t\t</node>\n'
    return initExtensions+initName+initIcolor+initOcolor+initFont+initPos+initBar+endExtensions

  def printProbabilities(self):
    commentProbabilities = '// setting probabilities for "'+self.name+'"\ntheProbs.Flush();\n'
    initProbabilities = 'theProbs.SetSize('+str(len(self.probability))+');\n'
    for i,probability in enumerate(self.probability):
      initProbabilities += 'theProbs['+str(i)+'] = '+str(probability)+';\n'
    endProbabilities = 'theNet.GetNode('+self.nodeId+')->Definition()->SetDefinition(theProbs);\n\n'
    print_prob = commentProbabilities+initProbabilities+endProbabilities
    return print_prob

  def getName(self):
    return self.name

  def getNodeId(self):
    return self.nodeId

  def setInteriorColor(self,interior_color):
    """Set interior color

    :Args:
      - interior_color (str): Interior color given as hex string

    :Example:
       >>>
       # red: rgb(255, 0, 0) -> ff0000
       xy.setInteriorColor('ff0000')
    """
    self.interior_color = interior_color

  def setOutlineColor(self,outline_color):
    """Set outline color

    :Args:
      - outline_color (str): Outline color given as hex string

    :Example:
       >>>
       # red: rgb(255, 0, 0) -> ff0000
       xy.setOutlineColor('ff0000')
    """
    self.outline_color = outline_color

  def setFontColor(self,font_color):
    """Set font color

    :Args:
      - font_color (str): Font color given as hex string

    :Example:
       >>>
       # red: rgb(255, 0, 0) -> ff0000
       xy.setFontColor('ff0000')
    """
    self.font_color = font_color

  def setFontName(self,font_name):
    """Set font name

    :Args:
      - font_name (str): Type of font, e.g. 'Arial', 'Times New Roman'
    """
    self.font_name = font_name

  def setFontSize(self,font_size):
    """Set font size

    :Args:
      - font_size (int): Size of font, e.g. 8, 12, ...
    """
    self.font_size = font_size

  def setNodeSize(self,x,y):
    """Set size of the node

    :Args:
      - x (int): Size in x direction
      - y (int): Size in y direction
    """
    self.node_size[2] = x
    self.node_size[3] = y

  def setNodePosition(self,x,y):
    """Set position of the node

    :Args:
      - x (int): Position in x direction
      - y (int): Position in y direction
    """
    self.node_position[0] = x
    self.node_position[1] = y
    self.node_position[2] = x+self.node_size[2]
    self.node_position[3] = y+self.node_size[3]

  def getNodePosition(self):
    return self.node_position

  def setBarActive(bar_active):
    """View node as Icon or Bar Chart

    :Args:
      - bar_active (boolean): ``True`` for Bar Chart view\n
        ``False`` for Icon view
    """
    self.bar_active = bar_active

class Arc(object):
  """Arc between two nodes

  :Attributes:
    - from_node (Node): Node where the arc starts
    - to_node (Node): Node where the arc points
  """

  def __init__(self, from_node, to_node):
    self.from_node = from_node
    self.to_node = to_node
    self.to_node.addArcConnection(self.from_node.getName(),self.from_node.getIdNum(),self.from_node.getSize())

  def __repr__(self):
    return self.name

def repEmptySpace(string):
  return string.replace(' ', '_')

def chunks(l, n):
  """ Yield successive n-sized chunks from l.
  """
  for i in xrange(0, len(l), n):
    yield l[i:i+n]
