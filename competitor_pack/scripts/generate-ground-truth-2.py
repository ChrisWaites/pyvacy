#!/usr/bin/python3

################################################################################
# This script generates ground truth data out of the pre-processed dataset and
# its meta-inromation object.

import csv
import json
from random import randint, uniform
from time import time
import sys

# Number of buckets for counting of numeric values (integers and floats)
NUM_BUCKETS = 100

# The code will print log message to console when processing each n-th row
# specified here.
NUM_ROWS_BETWEEN_LOG_MESSAGES = 10000

startedAt = time()

################################################################################
# Initialization.

print('Initialization...')

# Reads command-line arguments:
# - Input dataset;
# - Dictionary (meta-information) on the input dataset;
# - Ground truth;
_, inDataFilename, inDataSpecsFilename, outGroundTruth = sys.argv

# Opens necessary files.
inDataSpecs = json.load(open(inDataSpecsFilename))
inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
dataHeader = next(inData)

# We need the total count of data rows.
numRows = 0
for row in inData:
  numRows += 1
print('Total number of rows =', numRows)

# We need to select the base rows for each test.
baseRows = set()
for index in range(300):
  baseRows.add(randint(1, numRows))

# Generates test cases.
inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
dataHeader = next(inData)

testCases = [];

rowIndex = 0
for row in inData:
  rowIndex += 1
  if rowIndex not in baseRows: continue
  testCase = []
  for columnId in range(len(dataHeader)):
    # Skip some columns randomly, otherwise very few entries will satisfy to
    # such strict classes in most of the cases.
    if uniform(0, 1) < 0.90:
      continue
    column = [columnId]
    spec = inDataSpecs[dataHeader[columnId]]
    if spec['type'] == 'enum':
      column.append('enum')
      column.append(row[columnId])
      selected = {row[columnId]}
      maxval = spec['maxval']
      while True:
        new = str(randint(0, maxval))
        if new in selected: break
        else:
          selected.add(new)
          column.append(new)
    elif row[columnId] == '':
      column.append('number')
      column.append('')
    else:
      column.append('number')
      column.append(float(row[columnId]))
      column.append(uniform(0, float(spec['max']) - float(spec['min'])))
    testCase.append(column)
  testCases.append(testCase)

# Now, for each test case we need to actually collect its statistics.

inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
dataHeader = next(inData)

counts = []
for testCase in testCases:
  counts.append(0)

rowIndex = 0
for row in inData:
  rowIndex += 1
  if rowIndex % NUM_ROWS_BETWEEN_LOG_MESSAGES == 0:
    print(f'{rowIndex / 1000 :12}k rows processed')
  for testCaseId in range(len(testCases)):
    testCaseOk = True
    for column in testCases[testCaseId]:
      columnId = column[0]
      spec = inDataSpecs[dataHeader[columnId]]
      if spec['type'] == 'enum':
        if row[columnId] not in column[2:]:
          testCaseOk = False
          break
      else:
        if column[2] == '':
          if row[columnId] is not '':
            testCaseOk = False
            break
        elif row[columnId] == '':
          testCaseOk = False
          break
        else:
          d = abs(float(row[columnId]) - column[2])
          if d > column[3]:
            testCaseOk = False
            break
    if testCaseOk is True: counts[testCaseId] += 1

print('Output...')

outGT = csv.writer(open(outGroundTruth, 'w', newline=''), dialect='excel')

for testCaseId in range(len(testCases)):
  testCase = testCases[testCaseId]
  outGT.writerow(['@NEWCASE!', float(counts[testCaseId]) / numRows])
  for column in testCase:
    outGT.writerow(column)
