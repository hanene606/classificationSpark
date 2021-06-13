# classificationSpark
#!/usr/bin/env python
# coding: utf-8

# In[39]:


######### this is a basic Machine Learning Model with python and spark on jupyter ###########


#### pyspark is the Python API of Apache Spark. It’s the way we have to interact with the framework using Python. The installation is very simple. These are the steps:
##Install Java 8 or higher on your computer.
##Install Python (I recommend > Python 3.6 from Anaconda)
##Install PySpark: pip install spark   


import pyspark


# In[40]:


###check the installed version of Spark
print(pyspark.__version__)


# In[41]:


from pyspark.sql import SparkSession  ### a spark session 


# In[42]:



spark = SparkSession     .builder     .appName('Covid Data') \   ##### name of application 
    .getOrCreate()


# In[43]:


spark ###### get information from spark session


# In[44]:


####Loading the data into Spark##
    #####b.csv

base = (spark.read
          .format("csv")
          .option('header', 'true')
          .load("b.csv"))


# In[46]:


#### visualisation of 25 rows of the  base
base.show(25)


# In[52]:


base.toPandas()  ### interact with Pandas easily


# In[55]:


# How many rows we have
base.count()

# The names of our columns
base.columns


# In[48]:


### Types of our columns
base.dtypes


# In[ ]:


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


# In[ ]:


###Before modeling let’s do the usual splitting between training and testing:

(training_data, test_data) = transformed_data.randomSplit([0.8,0.2])


# In[ ]:



###We will be using a Random Forest Classifier with is a Supervised Machine Learning 
#### our target is the label : covid and the others are the features
#### we build the modele named rfcovid
from pyspark.ml.classification import RandomForestClassifier
rfcovid = RandomForestClassifier(labelCol='Covid', 
                            featuresCol='features',
                            maxDepth=5)


# In[ ]:


#####we fit the model with the train dataset

model = rfcovid.fit(training_data)


# In[ ]:


####This will give us something called a transformer. And finally, we predict using the test dataset:

predictions = model.transform(test_data)


# In[ ]:


# Evaluate our model
##we will use a basic metric called the accuracy
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol='Covid', 
    predictionCol='prediction', 
    metricName='accuracy')


# In[ ]:


#we to get the accuracy we do
accuracy = evaluator.evaluate(predictions)
print('Test Accuracy = ', accuracy) 


# In[ ]:


#### before you go 
spark.stop()  ## stoppin the spark session


# In[ ]:
