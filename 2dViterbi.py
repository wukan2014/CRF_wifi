# -*- coding: utf-8 -*-
import numpy as np
import math
N = 15 # row number of state matrix
M = 3  # colomn number of state matrix
Y = np.zeros((N,M)) # state matrix with all nodes
NumberOfStateWithANode = 2
weightForNodeFeature = 1  # lamda_k
weightForEdgeFeature = 1  # miu_k
viterbiMaxMatrix = [] # N+M+1 row, pow(2,Sd(d)) colomn
viterbiArgMaxMatrix = [] # N+M+1 row, pow(2,Sd(d)) colomn

def T_values(d):   # nodes states in Td
    td_values = []
    if d <= N:
        for row in range(d-1,-1,-1):
            col = d-1 - row
            if col < M:
                td_values.append(Y[row, col])
    else:
        for col in range(d-N,M):
            row = d-1 - col
            if row < N and row >= 0:
                td_values.append(Y[row, col])
    return np.matrix(td_values)

def T_nodes(d): # nodes in Td
    td_nodes = []
    if d <= N:
        for row in range(d-1,-1,-1):
            col = d-1 - row
            if col < M:
                td_nodes.append((row,col))
    else:
        for col in range(d-N,M):
            row = d-1 - col
            if row < N and row >= 0:
                td_nodes.append((row,col))
    return td_nodes

def nodeNumberInTd(node):
    if node[0] + node[1] < N-1:
        return node[1]
    else:
        offset = node[0] + node[1] - N + 1
        return node[1] - offset

def Ed(d): # edges between Td-1 and Td
    td_nodes = T_nodes(d-1)
    edges = []
    for node in td_nodes:
        edges.append((node,(node[0]+1,node[1])))
        edges.append((node, (node[0], node[1]+1)))
    return edges

def Sd(d): # get number of nodes in T_d
    if d == 0:
        return 0
    return len(T_nodes(d))

def state_with_node(node):
    return Y[node[0],node[1]]

def state_with_edge(edge):
    return (Y[edge[0][0],edge[0][1]],Y[edge[1][0],edge[1][1]])

def nodeFeatureFunction(node,x):
    return 1

def edgeFeatureFunction(edge,x):
    return 1

def Md_entry(Td_1assignment, Td_assignment,d,x): #Md(Td-1,Td |x)
    energy = 0
    if Td_1assignment == []:
        for node in T_nodes(d):
            energy += weightForNodeFeature * nodeFeatureFunction(Td_assignment[nodeNumberInTd(node[1])], x)
        return math.exp(energy)
    elif Td_assignment == []:
        return 1
    else:
        for edge in Ed(d):
            energy += weightForNodeFeature * edgeFeatureFunction(Td_1assignment[nodeNumberInTd(edge[0])], Td_assignment[nodeNumberInTd(edge[1])], x)
        for node in T_nodes(d):
            energy += weightForNodeFeature * nodeFeatureFunction(Td_assignment[nodeNumberInTd(node[1])], x)
        return math.exp(energy)

def decimalToNBaseByNormal(decimalVar, base,length):
    tempList = []
    temp = decimalVar
    i = 0
    while (temp > 0):
        ord = temp % base
        if (ord > 9): #如果余数大于9，则以字母的形式表示
            ord = chr(65 + (ord - 10))   #把数字转换成字符
        tempList.append(ord)
        temp = int(temp / base)
        i = i + 1
    tempList.reverse();
    #print(tempList)
    binary = []
    for j in range(length - len(tempList)):
        tempList.insert(0,0)
    for j in range(len(tempList)):
        # binary = binary + str(tempList[j]);
        binary.append(tempList[j])
    return binary
    # print("the decimal is: %d and after convering by %d base is %s"%(decimalVar, base, binary))

def assignmentsInTd(d):
    assignments = []
    for i in range(int(math.pow(NumberOfStateWithANode, Sd(d)))):
        assignments.append(decimalToNBaseByNormal(i, NumberOfStateWithANode, Sd(d)))
    return assignments

def numberOfAssignmentsInTd(d):
    return int(math.pow(NumberOfStateWithANode, Sd(d)))

# for d in range(0,N+M+1):
#     print T_nodes(d),Sd(d)
#     for node in T_nodes(d):
#         print nodeNumberInTd(node)


def Md_matrix(d,x):
    Md_matrix = np.zeros((math.pow(NumberOfStateWithANode, Sd(d - 1)) , math.pow(NumberOfStateWithANode, Sd(d))))
    for i, Td_1assignment in enumerate(assignmentsInTd(d-1)):
        for j, Td_assignment in enumerate(assignmentsInTd(d)):
            Md_matrix[i,j] = Md_entry(Td_1assignment, Td_assignment,d,x)
    return Md_matrix

def Z(x): # not sure what's (start,stop)entry
    Md = Md_matrix(0,x)
    for i in range(1,M+N-1):
        np.dot(Md,Md_matrix(i,x))
    return 1

def problity_givenX(assignments,x):
    lastAssignment = assignments[0]
    value = 1.0
    for d, assignment in enumerate(assignments[1:]):
        value *= Md_entry(lastAssignment,assignment,d+1,x)
        lastAssignment = assignment
    return value / Z(x)

def viterbi(x):
    viterbiArgMaxMatrix.append([])
    maxValues = []
    for assignment in assignmentsInTd(1):
        maxValues.append(Md_entry([], assignment, 1,x))
    viterbiMaxMatrix.append(maxValues)
    for d in range(2,N+M+1):
        maxValues = []
        argMax = [] # from which assignment
        for assignment in assignmentsInTd(d):
            tempMax = -1
            tempArgMax = -1
            for index,lastAssignment in enumerate(assignmentsInTd(d-1)) :
                currentValue = viterbiMaxMatrix[d-1][index]* Md_entry(lastAssignment,assignment,d,x)
                if currentValue > tempMax:
                    tempArgMax = (index,lastAssignment)
                    tempMax = currentValue
            maxValues.append(tempMax)
            argMax.append(tempArgMax)
        viterbiMaxMatrix.append(maxValues)
        viterbiArgMaxMatrix.append(argMax)
    optimimum_assignment = []
    lastIndex = viterbiArgMaxMatrix[M+N][0][0]
    lastAssignment = viterbiArgMaxMatrix[M+N][0][1]
    optimimum_assignment.insert(0, lastAssignment)
    for d in range(M+N-1,-1,-1):
        lastIndex = viterbiArgMaxMatrix[d][lastIndex][0]
        lastAssignment = viterbiArgMaxMatrix[d][lastIndex][1]
        optimimum_assignment.insert(0, lastAssignment)
    print viterbiMaxMatrix,viterbiArgMaxMatrix,optimimum_assignment
    return optimimum_assignment












