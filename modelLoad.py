#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession

session = SparkSession.builder.appName('Wine Quality Prediction Model Load').getOrCreate()

#Importing sys for taking the command line parameters
import sys
fileName = sys.argv[1]

#Load Random Forest Model package
from pyspark.mllib.tree import RandomForestModel
loadedRFModel = RandomForestModel.load(session.sparkContext,"myRandomForestClassificationModel")

data = session.read.format('csv').option('header','true').option('inferSchema','true').option('sep',';').load(fileName)

from pyspark.mllib.regression import LabeledPoint
modelData = data.rdd.map(lambda col: LabeledPoint(col[11],col[:11]))

predictionData = loadedRFModel.predict(modelData.map(lambda x: x.features))

labelAndPredictionData = modelData.map(lambda lp: lp.label).zip(predictionData)

#For F1 score using Random Forest with given dataset
from pyspark.mllib.evaluation import MulticlassMetrics
randomFResults = MulticlassMetrics(labelAndPredictionData)
randomFConfMatrix = randomFResults.confusionMatrix().toArray()
randomFPrecision = (randomFConfMatrix[0][0])/(randomFConfMatrix[0][0]+randomFConfMatrix[1][0])
randomFRecall = (randomFConfMatrix[0][0])/(randomFConfMatrix[0][0]+randomFConfMatrix[0][1])
randomFF1=(2*randomFPrecision*randomFRecall)/(randomFPrecision+randomFRecall)
print("=======================================================================================================")
print("F1 score using imported Random Forests model on the given dataset: " + str(randomFF1))
print("=======================================================================================================")
