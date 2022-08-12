import numpy as np
import csv
import pandas as pd
import math
import copy
from scipy import stats
from scipy.stats import chi2, norm

class Node:
    visited = False
    def __init__(self, data, rows, children=None, parent=None, branchName=''):
        self.data = data
        self.children = children or []
        self.parent = parent
        self.branchName = branchName
        self.rows = rows
    def addNode(self, data, branchName, rows):
        if (type(data) == type('str')):
            newNode = Node(data, rows, None, self, branchName)
            self.children.append(newNode)
        else:
            data.parent = self
            self.children.append(data)
    def printNode(self, toAdd):
        if (self.branchName == ''):
            print(toAdd, '|' + self.data + '|')
        elif len(self.children) == 0:
            print(toAdd, self.branchName, " => ", self.data)
        else:
            print(toAdd, self.branchName)
            print(toAdd + '--', '|' + self.data + '|')
        children = self.children
        if type(children) == type([]):
            for child in children:
              child.printNode(toAdd + '--')
    def deleteNode(self, value):
        self.children = []
        self.data = value
        return self


# Read CSV in pandas
restaurantCSV = pd.read_csv('restaurant.csv', header=None, skipinitialspace=True)

# Attribute list
columns = ['ALTERNATE', 'BAR', 'FRISAT', 'HUNGRY', 'PATRONS', 'PRICE', 'RAINING', 'RESERVATION', 'TYPE', 'WAITESTIMATE', 'RESULT']
restaurantCSV.columns = columns

restaurantCSV['RESULT'] = restaurantCSV['RESULT'].str.strip()
print('======== CSV file otuput ========')
print(restaurantCSV)


def pluralityValue(examples):
    yes_length = examples[examples['RESULT'] == 'Yes']
    no_length = examples[examples['RESULT'] == 'No']
    if len(yes_length) > len(no_length):
        return 'Yes'
    else:
        return 'No'

def isSameClassification(examples):
    if (len(examples[examples['RESULT'] == 'Yes']) == len(examples)) or (len(examples[examples['RESULT'] == 'No']) == len(examples)):
        return True
    else: return False


def getClassification(examples):
    if len(examples[examples['RESULT'] == 'Yes']) == len(examples):
        return 'Yes'
    elif len(examples[examples['RESULT'] == 'No']) == len(examples):
        return 'No'


# entropy formula is B(value) = -(value log2 (value)) + (1 - value) log2 (1 - value))
def entropy(value):
    if (value == 0 or value == 1):
        return 0
    return - (value * math.log(value, 2) + (1 - value) * math.log((1 - value), 2))

def remainder(attribute, examples):
    remainder = 0 
    unique_values = examples[attribute].unique()
    for value in unique_values:
        value_mask = examples[attribute] == value
        value_rows = examples[value_mask]
        yes_mask = value_rows['RESULT'] == 'Yes'
        remainder += (len(value_rows) / len(examples)) * (entropy(len(value_rows[yes_mask]) / len(value_rows)))
    return remainder


def informationGain(examples):
    informationGain = []
    bestAttr = ''
    bestVal = -np.inf
    for attr in examples.columns:
        if (attr != 'RESULT'):
            gain = 1 - remainder(attr, examples)
            informationGain.append(round(gain, 2))
            if (gain > bestVal):
                bestVal = gain
                bestAttr = attr
    return bestAttr, informationGain, bestVal

def isEmpty(examples):
    if len(examples) == 0:
        return True
    elif len(examples) == 1 and len(examples.columns) == 1:
        return True
    else:
        return False

def learnDecisionTree(examples, parent_examples, branch_value):
    if isEmpty(examples):
        return pluralityValue(parent_examples)
    elif isSameClassification(examples):
        classi = getClassification(examples)
        return classi
    elif len(examples.columns) == 1:
        return pluralityValue(parent_examples)
    else:
        best_attr, information_gain, best_val = informationGain(examples)
        if (best_attr):
            unique_values = parent_examples[best_attr].unique()
            parent_node = Node(best_attr, examples, None, None, branch_value)
            for value in unique_values:
                value_mask = examples[best_attr] == value
                value_rows = examples[value_mask]
                attribute_removed_rows = value_rows.drop(best_attr, axis=1)
                subtree = learnDecisionTree(attribute_removed_rows, parent_examples, value)
                parent_node.addNode(subtree, value, attribute_removed_rows)
            return parent_node


decisionTree = learnDecisionTree(restaurantCSV, restaurantCSV, '')
print('======== Decision Tree Output ========')
decisionTree.printNode('|')

def isAllLeafNodes(tree):
    child = tree.children
    allLeaves = 0
    for a in child:
        if len(a.children) == 0:
            allLeaves += 1
    if allLeaves == len(child):
        return True
    else: return False


def isLeafNode(node):
    if(node.children == []):
        return True
    else:
        return False

def pruneTree(tree, restaurantCSV):
    traverse_tree = copy.deepcopy(tree)
    queue = []
    queue.append(traverse_tree)
    chi_square_test_pass = False
    while len(queue) != 0:
        node = queue.pop(0)
        if not isLeafNode(node) and isAllLeafNodes(node):
            parent_pos_values = len(node.rows[node.rows['RESULT'] == 'Yes'])
            parent_neg_values = len(node.rows[node.rows['RESULT'] == 'No'])
            children = node.children
            distribution = 0
            for child in children:
                if (child.rows.empty):
                    continue
                children_pos_values = child.rows[child.rows['RESULT'] == 'Yes']
                children_neg_values = child.rows[child.rows['RESULT'] == 'No']
                expected_positive_values = parent_pos_values * (len(children_pos_values)/ len(node.rows))
                expected_negative_values = parent_neg_values * (len(children_pos_values)/ len(node.rows))
                pos_sq = (len(children_pos_values) - expected_positive_values) ** 2
                neg_sq = (len(children_neg_values) - expected_negative_values) ** 2
                positive_value = 0
                negative_value = 0
                if expected_positive_values == 0.0:
                    continue
                else:
                    positive_value = (pos_sq / expected_positive_values)
                if expected_negative_values == 0.0:
                    continue
                else:
                    negative_value = (neg_sq / expected_negative_values) 
                distribution += positive_value + negative_value
            critical_value = chi2.ppf(0.95, len(node.children) - 1)
            print(distribution, critical_value)
            if distribution < critical_value:
                maximum_frequency = 'No'
                if parent_pos_values > parent_neg_values:
                    maximum_frequency = 'Yes'
                node.deleteNode(maximum_frequency)
                chi_square_test_pass = True
        children = node.children
        for child in children:
            queue.append(child)
    return traverse_tree, chi_square_test_pass

print('======== Pruning Output ========')
pruned_tree = copy.deepcopy(decisionTree)
chi_square_test_pass = True
prune_count = 0
while chi_square_test_pass == True:
    prune_count += 1
    pruned_tree, chi_square_test_pass = pruneTree(pruned_tree, restaurantCSV)
    if chi_square_test_pass == True:
        print('Prune ', prune_count)
        pruned_tree.printNode('|')
