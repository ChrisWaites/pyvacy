#!/usr/bin/python3

################################################################################
# Local scorer.

import csv
import json
import sys
import copy
from math import log, sqrt

# The code will print log message to console when processing each n-th row
# specified here.
NUM_ROWS_BETWEEN_LOG_MESSAGES = 10000

# Reads command-line arguments:
# - Solution
# - Data Specs
# - Ground truth #1
# - Ground truth #2

_, inDataFilename, inDataSpecsFilename, inGroundTruth1, inGroundTruth2, inGroundTruth3 = sys.argv


################################################################################
# SCORING METHOD #1

def calcScore1():
  print('*********************************************************************')
  print('SCORING METHOD #1')

  ##############################################################################
  # Initialization.

  NUM_BUCKETS = 100

  print('Intitialization...')

  # Opens files.

  inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
  inGT = csv.reader(open(inGroundTruth1, newline=''), dialect='excel')
  inDataSpecs = json.load(open(inDataSpecsFilename))
  dataHeader = next(inData)

  # Loads ground truth data

  counts = {}
  columnSets = []

  for row in inGT:
    if row[0] == '@NEWCASE!':
      testId = len(columnSets)
      counts[testId] = {}
      columnSets.append([int(row[1]), int(row[2]), int(row[3])])
    else:
      test = counts[testId]
      key1 = row[0]
      key2 = row[1]
      key3 = row[2]
      count = row[3]
      if key1 != '': key1 = int(key1)
      if key2 != '': key2 = int(key2)
      if key3 != '': key3 = int(key3)
      if count != '': float(count)
      if key1 not in test: test[key1] = {}
      if key2 not in test[key1]: test[key1][key2] = {}
      if key3 not in test[key1][key2]: test[key1][key2][key3] = count

  ##############################################################################
  # Getting stats from the submission.

  print('Collecting stats from the solution...')

  solution = {}

  def countRow(row, bucket, cols, depth):
    value = row[cols[depth]]
    d = inDataSpecs[dataHeader[cols[depth]]]      
    if d['type'] == 'enum':
      if value == '': bucketId = 'outrange'
      else:
        bucketId = int(value)
        if bucketId < 0 or bucketId > d['maxval']: bucketId = 'outrange'
    elif value == '': bucketId = ''
    else: # integer or float
      bucketSize = 1.0001 * float(d['max'] - d['min']) / NUM_BUCKETS
      bucketId = int((float(value) - d['min']) / bucketSize)
      if bucketId < 0 or bucketId >= NUM_BUCKETS: bucketId = 'outrange'
    
    # bucketId = 'outrange'

    if depth == 2: # We are at leaf > just increment the counter.
      if bucketId in bucket: bucket[bucketId] += 1
      else: bucket[bucketId] = 1
    else:
      if bucketId not in bucket: bucket[bucketId] = {}
      bucket[bucketId] = countRow(row, bucket[bucketId], cols, 1 + depth)

    return bucket

  rowIndex = 0
  for row in inData:
    # Logging
    rowIndex += 1
    if rowIndex % NUM_ROWS_BETWEEN_LOG_MESSAGES == 0:
      print(f'{rowIndex / 1000 :12}k rows processed')

    # if rowIndex == 10000: break

    # Count the row for each ground truth chunk
    for index in range(len(columnSets)):
      if index not in solution: solution[index] = {}
      cset = columnSets[index]
      if cset[2] < len(row) and cset[1] < len(row) and cset[0] < len(row):
        solution[index] = countRow(row, solution[index], cset, 0)
      else: solution[index] = { 'outrange': { 'outrange': { 'outrange': rowIndex } } }

  ##############################################################################
  # Scoring.

  print('Scoring...')

  score = len(counts.keys())

  for testId in counts:
    if testId not in solution: score -= 1.0
    else:
      for key1 in solution[testId]:
        for key2 in solution[testId][key1]:
          for key3 in solution[testId][key1][key2]:
            v1 = float(solution[testId][key1][key2][key3]) / rowIndex
            if key1 in counts[testId] and key2 in counts[testId][key1] and key3 in counts[testId][key1][key2]:
              v2 = float(counts[testId][key1][key2][key3])
              score += v2
            else: v2 = 0
            score -= abs(v2 - v1)

  score *= 1000000.0 / 2 / len(counts.keys())

  print(f'SCORE #1 = {score}')
  return score


################################################################################
# SCORING METHOD #2

def calcScore2():

  print('*********************************************************************')
  print('SCORING METHOD #2')

  ##############################################################################
  # Initialization.

  inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
  inGT = csv.reader(open(inGroundTruth2, newline=''), dialect='excel')
  dataHeader = next(inData)

  # Loads ground truth data.

  testCases = []
  testCase = None
  for row in inGT:
    if row[0] == '@NEWCASE!':
      if testCase != None: testCases.append(testCase)
      testCase = [float(row[1])]
    else:
      item = [int(row[0]), row[1]]
      if row[1] == 'enum':
        for x in row[2:]:
          item.append(int(x))
      else:
        if (row[2] == ''): item.append('')
        else:
          item.append(float(row[2]))
          item.append(float(row[3]))
      testCase.append(item)
  testCases.append(testCase)

  # Collects statistics from the solution
  counts = []
  for testCase in testCases:
    counts.append(0)

  numRows = 0
  for row in inData:
    numRows += 1
    if numRows % NUM_ROWS_BETWEEN_LOG_MESSAGES == 0:
      print(f'{numRows / 1000 :12}k rows processed')
    for testCaseId in range(len(testCases)):
      count = True
      for item in testCases[testCaseId][1:]:
        columnId = item[0]
        itemType = item[1]
        if columnId >= len(row):
          pass
        elif itemType == 'enum':
          if int(row[columnId]) not in item[2:]:
            count = False
            break
        else:
          if item[2] == '':
            if row[columnId] != '':
              count = False
              break
          elif row[columnId] == '':
            count = False
            break
          else:
            d = abs(float(row[columnId]) - item[2])
            if d > item[3]:
              count = False
              break
      if count is True: counts[testCaseId] += 1

  sum2 = 0
  for index in range(len(testCases)):
    x = log(max(counts[index] / numRows, 1e-6))
    gt = log(testCases[index][0])
    sum2 += (x - gt) * (x - gt)

  score = 1e6 * max(0, (1 + sqrt(sum2 / len(testCases)) / log(1e-3)))
  print(f'SCORE #2 = {score}')
  return score


################################################################################
# SCORING METHOD #3

def calcScore3():
  print('*********************************************************************')
  print('SCORING METHOD #3')

  ##############################################################################
  # Initialization.

  inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
  inGT = csv.reader(open(inGroundTruth3, newline=''), dialect='excel')
  dataHeader = next(inData)

  # Load ground truth data.
  gt = []
  for row in inGT:
    gt.append([int(row[0]), float(row[1]), float(row[2])])

  def sortThird(val):
    return -val[2]

  gt.sort(key = sortThird)

  data = {}

  if 'SEX' not in dataHeader or 'CITY' not in dataHeader or 'INCWAGE' not in dataHeader:
    print(f'SCORE #3 = {0}')
    return 0

  SEX_INDEX = dataHeader.index('SEX')
  CITY_INDEX = dataHeader.index('CITY')
  WAGE_INDEX = dataHeader.index('INCWAGE')

  # Collecting data
  print('Loading data...')
  rowIndex = 0
  for row in inData:
    rowIndex += 1
    if rowIndex % NUM_ROWS_BETWEEN_LOG_MESSAGES == 0:
      print(f'{rowIndex / 1000 :12}k rows processed')

    sex = int(row[SEX_INDEX]) - 1
    city = row[CITY_INDEX]
    wage = int(row[WAGE_INDEX])
    if city not in data:
      data[city] = []
    data[city].append([sex, wage])

  print('Calculating Gini indices...')

  def sortSecond(val):
    return val[1]

  results = {}

  for city in data:
    d = data[city]
    d.sort(key = sortSecond)
    totalPeople = 0
    totalWage = 0
    for item in d:
      if (item[1] < 999998): # 999998 is N/A code
        totalWage += item[1]
        totalPeople += 1
    g = 0
    accWage = 0
    if (totalPeople > 0.0 and totalWage > 0.0):
      for item in d:
        if (item[1] < 999998):
          accWage += item[1]
          g += accWage / totalPeople / totalWage
    results[city] = { 'gini': 1 - 2 * g }

  paygap = {}
  for city in data:
    d = data[city]
    buf = [[0, 0], [0, 0]]
    for item in d:
      if (item[1] < 999998): # 999998 is N/A code
        buf[item[0]][0] += 1
        buf[item[0]][1] += item[1]
    results[city]['paygap'] = 0;
    if (buf[0][0] > 0):
      results[city]['paygap'] += buf[0][1] / buf[0][0]
    if (buf[1][0] > 0):
      results[city]['paygap'] -= buf[1][1] / buf[1][0]

  sol = []
  for city in results:
    sol.append([int(city), float(results[city]['gini']), float(results[city]['paygap'])])
  sol.sort(key = sortThird)
  
  i = 0
  sol2 = {}
  for item in sol:
    sol2[item[0]] = { 'i': i, 'gini': item[1] }
    i += 1

  i = 0
  gt2 = {}
  for item in gt:
    gt2[item[0]] = { 'i': i, 'gini': item[1] }
    i += 1
  
  s1 = 0
  s2 = 0
  for key in gt2:
    if key in sol2:
      tmp = 1 - abs(gt2[key]['gini'] - sol2[key]['gini'])
      s1 += tmp * tmp
      tmp = 1 - abs(gt2[key]['i'] - sol2[key]['i']) / (i - 1)
      if (tmp < 0): tmp = 0.0
      s2 += tmp * tmp
  s1 = 500000 * sqrt(s1 / i)
  s2 = 500000 * sqrt(s2 / i)

  print(f'SCORE #3 = {s1 + s2}')
  return s1 + s2


################################################################################
# Score aggregation.

score1 = calcScore1()
score2 = calcScore2()
score3 = calcScore3()

score = (score1 + score2 + score3) / 3

print('OVERAL SCORE =', score)
