## Pedram Maleki CMSC416  Semtiment Analysis Prof: Dr. McInnes
## This program will use a training data to build a model for
## sentiment analysis of Tweets. The model is then used to find
## the sentiments of the test data. The model that is learned from
## training will be outputted to a file and the resulting analysis
## of the test data will be put to STOUT.
## the program runs as follows:
## python sentiment.py sentiment-train.txt sentiment-test.txt my-model.txt > my-sentiment-answers.txt
## This will write the putput of the stout to a file calle my-model-answers.txt
## in the following format:
## <answer instance="620979391984566272" sentiment="negative"/>
##
## the model is a bag of words feature.
##
## The accuracy of the model is 69.4%
## The confusion Matrix is as follows:
##
##  Predict        negative       positive
##  Actual
##  negative       2              70
##  positive       1              159
##  Most Frequent baseline is 160/232=68.9% which is lower than the calculated baseline


import math
import sys
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords#needs a pip install
nltk.download('stopwords')#needed for the tokenizer

#just removing extra chars from the sentences
def removeExtras(line):
    line = line.replace('<s>', '')
    line = line.replace('</s>', '')
    line = line.replace('<@>', '')
    # line=line.replace('<head>','')
    # line=line.replace('</head>','')
    line = line.replace('<p>', '')
    line = line.replace('</p>', '')
    line = line.replace('\"','')
    line = line.replace('.', '')
    line = line.replace('?','')
    line = line.replace('(','')
    line = line.replace(')','')
    line = line.replace(',', '')
    line = line.replace('--','')

    return line

#this method is used to delete the stop words from a sentence
def removeStopWords(line):
    stop_words = set(stopwords.words('english'))
    word_tokens = line.split()
    lineWithOutStopWords = [w for w in word_tokens if not w in stop_words]
    lineWithOutStopWords = []
    for w in word_tokens:
        if w not in stop_words:
            lineWithOutStopWords.append(w)

    return lineWithOutStopWords

#this method will rank the features and put them in a dictionary by their
#loglikelyhood score from high to low
def rankSortFeatures(allFeaturesDict, countPositiveTotal, countNegativeTotal, dominantSentiment):

    #calculating logLikelyhood and adding to a dict
    featuresRanked=defaultdict(lambda: defaultdict(float))


    #iterating through the feature that were extracted
    for word, tag in allFeaturesDict.items():

        #getting counts
        countPositive=tag['positive']
        countNegative=tag['negative']
        sentimentOfFeature=''


        #if the count of any sense is 0, it is added 0.1
        #to make sure there is no 0 denominator or numerator
        if countPositive == 0:
            countPositive = countPositive + 0.1
        if countNegative == 0:
            countNegative = countNegative + 0.1

    #     #getting the dominant sense. if they are equal the majority
    #     #sense is assigned
        if countPositive > countNegative:
            sentimentOfFeature='positive'
        if countPositive < countNegative:
            sentimentOfFeature='negative'
        if countNegative==countPositive:
            sentimentOfFeature=dominantSentiment


        #adding to a dict with two keys and the LL as the value
        featuresRanked[word][sentimentOfFeature]=abs(math.log((countPositive/countPositiveTotal)/(countNegative/countNegativeTotal)))

    # someDict=dict(featuresRanked)
    # for word, tag in someDict.items():
    #     print("\nword:", word)
    #     for key in tag:
    #         print(key + ':', tag[key])

    #changing from default dict to dict for easier use
    #iterating through and making a singe key for easier sorting
    featuresRankedDict=defaultdict()
    someDict=dict(featuresRanked)
    for word, tag in someDict.items():
        #print("\nword:", word)
        for key in tag:
            featuresRankedDict[word, key]=tag[key]
            #print(word ,key + ':', tag[key])
    #using the sorted method to sort the dict in reverse order which is
    #high to low
    aDict=dict(featuresRankedDict)
    sortedFeatures=sorted(aDict.items(), key=lambda feature: (feature[1], feature[0]), reverse=True)
    return sortedFeatures

def main():
    #command line args
    trainFileName=sys.argv[1]
    testFileName=sys.argv[2]
    myModelName=sys.argv[3]

    #creating a default dict for better performance
    allFeaturesDict = defaultdict(lambda:{'positive':0,'negative':0})
    sentimentCountDict = defaultdict(int)

    testInstanceWithSentimentDict = defaultdict()

    dominantSentiment=''

    toBeRemoved = []
    toBeRemovedFromTest = []

    trainFile = open(trainFileName, 'r')
    #processing train file
    while True:

        # Get next line from file
        line = trainFile.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break


        #<answer instance="620821002390339585" sentiment="negative"/>
        #using regex to find the target sentence
        instanceIDandSentiment=re.findall(r'<answer instance=(\".*\") sentiment=(\".*\")\/>',line)
        #if sentence is found by the regex the list is not0 and go in
        if len(instanceIDandSentiment)!=0:
            #just extracting the info
            tempString=instanceIDandSentiment[0][1]
            sentiment=str(tempString)
            sentiment=sentiment.replace('"','')
            # sense counter dict
            sentimentCountDict[sentiment] += 1
            #skipping a line to get to the sentence
            line = trainFile.readline()
            line = trainFile.readline()
            #casting into a string
            theTweet=str(line)
            # removing noise
            theTweet = removeExtras(theTweet)
            ########################TRY to remve @sign and maybe extras##################################
            ########################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^##################################
            theTweetTokens = removeStopWords(theTweet)

            #############################need to remove double links###############
            #going through the tokens to look for URL links
            for tokens in theTweetTokens:
                urlLink1 = re.findall(r'(.*)http(.*)',str(tokens))
                if urlLink1:
                    #theTweetTokens.remove(tokens)
                    toBeRemoved.append(tokens)
            #removing URL from the list
            theTweetTokens=[x for x in theTweetTokens if x not in toBeRemoved]

            #adding the tokens with the sentiment count to the dictionary
            for tokens in theTweetTokens:
                allFeaturesDict[tokens][sentiment] += 1

        else:
            #go to next line of the text file
            continue
    trainFile.close()

    #gettting the counts and calculating the majority sense
    countPositiveTotal = sentimentCountDict['positive']
    countNegativeTotal = sentimentCountDict['negative']
    if countPositiveTotal > countNegativeTotal:
        dominantSentiment='positive'

    if countPositiveTotal < countNegativeTotal:
        dominantSentiment='negative'

    # sorting features
    sortedFeatures = rankSortFeatures(allFeaturesDict, countPositiveTotal, countNegativeTotal, dominantSentiment)
    #print(sortedFeatures[1])

    #writing to the my-model file the model that was generated with log probablities
    myModelFile = open(myModelName,'w+')
    for features in sortedFeatures:
        myModelFile.write(str(features) +'\n')
    myModelFile.close()

    ########################################################################################################
    ##################################        TEST FILE PROCESSING     #####################################
    ########################################################################################################
    testFile = open(testFileName, 'r')

    # processing train file
    while True:

        # Get next line from file
        line = testFile.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break
        #looking for the line with instance ID
        instanceIDLine = re.search(r'<instance id=\"(.*)\">', line)
        #if found, store in the instanceID var
        if instanceIDLine:
            instanceID = str(instanceIDLine.group(1))

            #skipping a line to get to the sentence
            line = testFile.readline()
            line = testFile.readline()

            #casting into a string
            testTweet=str(line)
            # removing noise
            testTweet = removeExtras(testTweet)
            ########################TRY to remve @sign and maybe extras##################################
            ########################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^##################################
            testTweetTokens = removeStopWords(testTweet)

            #############################need to remove double links###############
            #going through the tokens to look for URL links
            for tokens in testTweetTokens:
                urlLink1 = re.findall(r'(.*)http(.*)',str(tokens))
                if urlLink1:
                    #theTweetTokens.remove(tokens)
                    toBeRemovedFromTest.append(tokens)
            #removing URL from the list
            testTweetTokens=[x for x in testTweetTokens if x not in toBeRemovedFromTest]

            #looping through sorted features
            for test in sortedFeatures:
                #turning the tuple into a string for easier processing
                temp=str(test)
                temp=temp.replace('(','')
                temp=temp.replace(')','')
                temp=temp.replace('\'','')
                temp=temp.replace(',','')
                temp1=temp.split()
                testWord=temp1[0]
                testSentiment=temp1[1]

                #looping through words in test Tweet
                for token in testTweetTokens:
                    if str(token) == testWord:
                        testInstanceWithSentimentDict[instanceID]=testSentiment
                    else:
                        continue
    #printing to the console,stdout
    someDict=dict(testInstanceWithSentimentDict)
    for id, sentiment in someDict.items():
        print('<answer instance=\"'+id+'\"'+' sentiment=\"'+sentiment+'\"/>')


if __name__ == "__main__":
    main()