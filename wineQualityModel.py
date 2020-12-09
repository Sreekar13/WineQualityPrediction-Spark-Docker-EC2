#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession

session = SparkSession.builder.appName('Wine Quality Prediction').getOrCreate()

#TrainingDataset.csv
trainData = session.read.format('csv').option('header','true').option('inferSchema','true').option('sep',';').load('TrainingDataset.csv')

transformedTrainData = trainData.withColumnRenamed('"""""fixed acidity""""','fixed acidity').withColumnRenamed('""""volatile acidity""""','volatile acidity').withColumnRenamed('""""citric acid""""','citric acid').withColumnRenamed('""""residual sugar""""','residual sugar').withColumnRenamed('""""chlorides""""','chlorides').withColumnRenamed('""""free sulfur dioxide""""','free sulfur dioxide').withColumnRenamed('""""total sulfur dioxide""""','total sulfur dioxide').withColumnRenamed('""""density""""','density').withColumnRenamed('""""pH""""','pH').withColumnRenamed('""""sulphates""""','sulphates').withColumnRenamed('""""alcohol""""','alcohol').withColumnRenamed('""""quality"""""','quality')


#Logistic Regression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint


lrTrainData = transformedTrainData.rdd.map(lambda col: LabeledPoint(col[11],col[:11]))


model = LogisticRegressionWithLBFGS.train(lrTrainData,numClasses=10)


qLabelWithQPredictionsTrain = lrTrainData.map(lambda pre: (pre.label, model.predict(pre.features)))


qLabelWithQPredictionsTrain = qLabelWithQPredictionsTrain.map(lambda a:(a[0],float(a[1])))


#ValidationDataset.csv
validationData = session.read.format('csv').option('header','true').option('inferSchema','true').option('sep',';').load('ValidationDataset.csv')


transformedValidationData = validationData.withColumnRenamed('"""""fixed acidity""""','fixed acidity').withColumnRenamed('""""volatile acidity""""','volatile acidity').withColumnRenamed('""""citric acid""""','citric acid').withColumnRenamed('""""residual sugar""""','residual sugar').withColumnRenamed('""""chlorides""""','chlorides').withColumnRenamed('""""free sulfur dioxide""""','free sulfur dioxide').withColumnRenamed('""""total sulfur dioxide""""','total sulfur dioxide').withColumnRenamed('""""density""""','density').withColumnRenamed('""""pH""""','pH').withColumnRenamed('""""sulphates""""','sulphates').withColumnRenamed('""""alcohol""""','alcohol').withColumnRenamed('""""quality"""""','quality')


lrValidationData = transformedValidationData.rdd.map(lambda col: LabeledPoint(col[11],col[:11]))


qLabelWithQPredictionsValidation = lrValidationData.map(lambda pre: (pre.label, model.predict(pre.features)))


qLabelWithQPredictionsValidation = qLabelWithQPredictionsValidation.map(lambda a:(a[0],float(a[1])))


#For F1 score on validation data set
from pyspark.mllib.evaluation import MulticlassMetrics

results = MulticlassMetrics(qLabelWithQPredictionsValidation)


confMatrix = results.confusionMatrix().toArray()


precision = (confMatrix[0][0])/(confMatrix[0][0]+confMatrix[1][0])
recall = (confMatrix[0][0])/(confMatrix[0][0]+confMatrix[0][1])
f1=(2*precision*recall)/(precision+recall)


print("F1 score of Logistic Regression on Validation Dataset: "+ str(f1))


#Trying RandomForest Model
from pyspark.mllib.tree import RandomForest, RandomForestModel


randomForestModel = RandomForest.trainClassifier(lrTrainData, numClasses = 10, categoricalFeaturesInfo={},featureSubsetStrategy="auto", numTrees=100)


randomForestPredictionsTrain = randomForestModel.predict(lrTrainData.map(lambda x: x.features))


randomForestlabelsAndPredictionsTrain = lrTrainData.map(lambda lp: lp.label).zip(randomForestPredictionsTrain)


#For F1 score using Random Forest with training data set
from pyspark.mllib.evaluation import MulticlassMetrics
randomFResults = MulticlassMetrics(randomForestlabelsAndPredictionsTrain)
randomFConfMatrix = randomFResults.confusionMatrix().toArray()
randomFPrecision = (randomFConfMatrix[0][0])/(randomFConfMatrix[0][0]+randomFConfMatrix[1][0])
randomFRecall = (randomFConfMatrix[0][0])/(randomFConfMatrix[0][0]+randomFConfMatrix[0][1])
randomFF1=(2*randomFPrecision*randomFRecall)/(randomFPrecision+randomFRecall)
print(randomFF1)


#Checking Random Forest on ValdiationDataset.csv
randomForestPredictionsValidation = randomForestModel.predict(lrValidationData.map(lambda x: x.features))


randomForestlabelsAndPredictionsValidation = lrValidationData.map(lambda lp: lp.label).zip(randomForestPredictionsValidation)


#For F1 score using Random Forest with training data set
from pyspark.mllib.evaluation import MulticlassMetrics
randomFResults = MulticlassMetrics(randomForestlabelsAndPredictionsValidation)
randomFConfMatrix = randomFResults.confusionMatrix().toArray()
randomFPrecision = (randomFConfMatrix[0][0])/(randomFConfMatrix[0][0]+randomFConfMatrix[1][0])
randomFRecall = (randomFConfMatrix[0][0])/(randomFConfMatrix[0][0]+randomFConfMatrix[0][1])
randomFF1=(2*randomFPrecision*randomFRecall)/(randomFPrecision+randomFRecall)
print("F1 score using Random Forests algorithm model trained on TrainingDataset.csv to ValidationDataset.csv: " + str(randomFF1))


randomForestModel.save(session.sparkContext, "target/myRandomForestClassificationModel")

