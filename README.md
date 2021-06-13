# classificationSpark
#!/usr/bin/env python
# coding: utf-8




######### this is a basic Machine Learning Model with python and spark on jupyter ###########


#### pyspark is the Python API of Apache Spark. It’s the way we have to interact with the framework using Python. The installation is very simple. These are the steps:
##Install Java 8 or higher on your computer.
##Install Python (I recommend > Python 3.6 from Anaconda)
##Install PySpark: pip install spark   


import pyspark





###check the installed version of Spark
print(pyspark.__version__)




from pyspark.sql import SparkSession  ### a spark session 





spark = SparkSession     .builder     .appName('Covid Data') \   ##### name of application 
    .getOrCreate()



###### get information from spark session
spark

####Loading the data into Spark##
    #####b.csv

base = (spark.read
          .format("csv")
          .option('header', 'true')
          .load("b.csv"))





#### visualisation of 25 rows of the  base
base.show(25)



base.toPandas()  ### interact with Pandas easily





# How many rows we have
base.count()

# The names of our columns
base.columns





### Types of our columns
base.dtypes




####Building A Machine Learning Model With Spark ML#####


###As you can see the new column features contain the same information from all of our features but in a list-like object. To do that in Spark we use the VectorAssembler:
required_features = ['Sexe',
                    'Age',
                    'Poid',
                    'Taille',
                     'Ville',
                     'Maladies chroniques'
                   ]
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=required_features, outputCol='features')
transformed_data = assembler.transform(base)




###Before modeling let’s do the usual splitting between training and testing:

(training_data, test_data) = transformed_data.randomSplit([0.8,0.2])






###We will be using a Random Forest Classifier with is a Supervised Machine Learning 
#### our target is the label : covid and the others are the features
#### we build the modele named rfcovid
from pyspark.ml.classification import RandomForestClassifier
rfcovid = RandomForestClassifier(labelCol='Covid', 
                            featuresCol='features',
                            maxDepth=5)




#####we fit the model with the train dataset

model = rfcovid.fit(training_data)




####This will give us something called a transformer. And finally, we predict using the test dataset:

predictions = model.transform(test_data)





# Evaluate our model
##we will use a basic metric called the accuracy
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol='Covid', 
    predictionCol='prediction', 
    metricName='accuracy')





#we to get the accuracy we do
accuracy = evaluator.evaluate(predictions)
print('Test Accuracy = ', accuracy) 





#### before you go 
spark.stop()  ## stoppin the spark session



