############################################################
##  Pedram Maleki   CMSC-416    Dr. McInnes
##  PA5 This program is used to calculate the overall
##  accuracy of the sentiment.py program. It is also used to
##  create a confusion matrix where it will show what sentiment
##  were in the wrong place.
##
##
##  Both input files are processed and stored in lists.
##  Then we iterate through the lists and check them
##  one by one to see if they match
##  At the end a confusion matrix is creatde using the pycm
##  library
##
############################################################
############################################################
#####################----NOTE----###########################
######                                       ###############
######  In order to use the file you have to ###############
######  do a pip install of pycm --> pip install pycm ######
############################################################
############################################################

import sys
import re
from collections import defaultdict
#import pandas as pd
from pycm import *

#main function
def main():
    #input from command line
    testFileName=sys.argv[1]
    keyFileName=sys.argv[2]

    #a default dictionary is used to avoid having to
    #check for keys already in the dictionary and
    #if not adding and also for better performance
    testFileDict = defaultdict()
    keyFileDict = defaultdict()

    #two lists that will hold the sentiments
    keySentimentList = []
    testSentimentList = []


    #reading the file
    testFile = open(testFileName, 'r')

    while True:

        # Get next line from file
        line = testFile.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break
        line=line.replace('\"','')
        temp = re.findall(r'<answer instance=(.*)sentiment=(.*)/>', line)
        instanceId=str(temp[0][0]).strip()
        testSentiment=str(temp[0][1])
        #adding to the dictionary
        testFileDict[instanceId]=testSentiment


    testFile.close()

    # Using readline()
    keyFile = open(keyFileName, 'r')

    while True:

        # Get next line from file
        line = keyFile.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break
        #processing the line and extarcting the ID and sentiment
        line=line.replace('\"','')
        temp = re.findall(r'<answer instance=(.*)sentiment=(.*)/>', line)
        instanceId=str(temp[0][0]).strip()
        keySentiment=str(temp[0][1]).strip()
        #adding to the key dict
        keyFileDict[instanceId]=keySentiment



    keyFile.close()

    #converting to regular dicts for easier processecing
    testFileDict1=dict(testFileDict)
    keyFileDict1=dict(keyFileDict)
    correctCounter=0


    #looping through the test file and checking the key
    #in the key file for equality.
    for id, sentmnt in testFileDict1.items():

        testFileSentiment=sentmnt
        keyFileSentiment=keyFileDict1[id]
        #if the values are equal we have a correct sentiment
        #increment the correct counter
        if str(testFileSentiment) == str(keyFileSentiment):
            correctCounter +=1

    #adding to lists to produce the confusion matrix
    for word in keyFileDict1:
        keySentimentList.append(keyFileDict1[word])

    for word in testFileDict1:
        testSentimentList.append(testFileDict1[word])
    #calculating the accuracy of the model
    countTotal=len(keySentimentList)
    accuracy=round(100*(correctCounter/countTotal),2)
    print('Accuracy---->>>:',accuracy)



    #using pycm library the two lists are passed to the confusion matrix
    #method and then are printed.
    cm = ConfusionMatrix(actual_vector=keySentimentList, predict_vector=testSentimentList)
    cm.classes
    cm.print_matrix()
    print(cm)


if __name__ == "__main__":
    main()