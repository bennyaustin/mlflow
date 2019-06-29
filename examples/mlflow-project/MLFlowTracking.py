# Databricks notebook source
# MAGIC %md Adapted From https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/azure-databricks/Databricks_AMLSDK_1-4_6.dbc which happens to be an AML example !

# COMMAND ----------

# MAGIC %md #Data Ingestion

# COMMAND ----------

import os
import urllib

# COMMAND ----------

# DBTITLE 1, Download AdultCensusIncome.csv from Azure CDN. This file has 32,561 rows.
# Download AdultCensusIncome.csv from Azure CDN. This file has 32,561 rows.
if __name__ == "__main__":
  basedataurl = "https://amldockerdatasets.azureedge.net"
  datafile = "AdultCensusIncome.csv"
  datafile_dbfs = os.path.join("/dbfs", datafile)

  if os.path.isfile(datafile_dbfs):
      print("found {} at {}".format(datafile, datafile_dbfs))
  else:
      print("downloading {} to {}".format(datafile, datafile_dbfs))
      urllib.request.urlretrieve(os.path.join(basedataurl, datafile), datafile_dbfs)

# COMMAND ----------

# Create a Spark dataframe out of the csv file.
# if __name__ == "__main__":
  data_all = sqlContext.read.format('csv').options(header='true', inferSchema='true', ignoreLeadingWhiteSpace='true', ignoreTrailingWhiteSpace='true').load(datafile)
  print("({}, {})".format(data_all.count(), len(data_all.columns)))
  data_all.printSchema()

# COMMAND ----------

#renaming columns
# if __name__ == "__main__":
  columns_new = [col.replace("-", "_") for col in data_all.columns]
  data_all = data_all.toDF(*columns_new)
  data_all.printSchema()

# COMMAND ----------

#ensure that you see a table with 5 rows and various columns
# if __name__ == "__main__":
  display(data_all.limit(5))

# COMMAND ----------

# MAGIC %md #Data Preparation

# COMMAND ----------

# Choose feature columns and the label column.

# if __name__ == "__main__":
  label = "income"
  xvars = set(data_all.columns) - {label}

  print("label = {}".format(label))
  print("features = {}".format(xvars))

  data = data_all.select([*xvars, label])

  # Split data into train and test.
  train, test = data.randomSplit([0.80, 0.20], seed=13)

  print("train ({}, {})".format(train.count(), len(train.columns)))
  print("test ({}, {})".format(test.count(), len(test.columns)))

# COMMAND ----------

# MAGIC %md #Data Persistence

# COMMAND ----------

# Write the train and test data sets to intermediate storage
# if __name__ == "__main__":
  train_data_path = "AdultCensusIncomeTrain"
  test_data_path = "AdultCensusIncomeTest"

  train_data_path_dbfs = os.path.join("/dbfs", "AdultCensusIncomeTrain")
  test_data_path_dbfs = os.path.join("/dbfs", "AdultCensusIncomeTest")

  train.write.mode('overwrite').parquet(train_data_path)
  test.write.mode('overwrite').parquet(test_data_path)
  print("train and test datasets saved to {} and {}".format(train_data_path_dbfs, test_data_path_dbfs))

# COMMAND ----------

# MAGIC %md #Model Building#

# COMMAND ----------

# if __name__ == "__main__":
  import pprint
  import numpy as np

  from pyspark.ml import Pipeline, PipelineModel
  from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
  from pyspark.ml.classification import LogisticRegression
  from pyspark.ml.evaluation import BinaryClassificationEvaluator
  from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

#get the train and test datasets
# if __name__ == "__main__":
  train = spark.read.parquet(train_data_path)
  test = spark.read.parquet(test_data_path)

  print("train: ({}, {})".format(train.count(), len(train.columns)))
  print("test: ({}, {})".format(test.count(), len(test.columns)))

  train.printSchema()

# COMMAND ----------

# MAGIC %md #Define Model#

# COMMAND ----------

# if __name__ == "__main__":
  label = "income"
  dtypes = dict(train.dtypes)
  dtypes.pop(label)

  si_xvars = []
  ohe_xvars = []
  featureCols = []
  for idx,key in enumerate(dtypes):
      if dtypes[key] == "string":
          featureCol = "-".join([key, "encoded"])
          featureCols.append(featureCol)
        
          tmpCol = "-".join([key, "tmp"])
          # string-index and one-hot encode the string column
          #https://spark.apache.org/docs/2.3.0/api/java/org/apache/spark/ml/feature/StringIndexer.html
          #handleInvalid: Param for how to handle invalid data (unseen labels or NULL values). 
          #Options are 'skip' (filter out rows with invalid data), 'error' (throw an error), 
          #or 'keep' (put invalid data in a special additional bucket, at index numLabels). Default: "error"
          si_xvars.append(StringIndexer(inputCol=key, outputCol=tmpCol, handleInvalid="skip"))
          ohe_xvars.append(OneHotEncoder(inputCol=tmpCol, outputCol=featureCol))
      else:
          featureCols.append(key)

  # string-index the label column into a column named "label"
  si_label = StringIndexer(inputCol=label, outputCol='label')

  # assemble the encoded feature columns in to a column named "features"
  assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

# COMMAND ----------

# DBTITLE 1,Initialize MLflow
# if __name__ == "__main__":
  import mlflow
  from mlflow.tracking import *

  #Instantiate MlflowClient
  mlflowclient= MlflowClient() 

  experiment_name="/Shared/MLFlow/IncomePrediction"
  artifacts_folder="IncomePrediction"

  secret_scope="dbvault"
  storage_account="storageg2"
  storage_access_key= dbutils.secrets.get(secret_scope,"storageg2_key")
  artifact_mountpoint="/mnt/artifact"

  if not any(mount.mountPoint ==artifact_mountpoint for mount in dbutils.fs.mounts()):
    dbutils.fs.mount( source = "wasbs://ml-artifacts@" + storage_account + ".blob.core.windows.net",
            mount_point = artifact_mountpoint,
            extra_configs = {"fs.azure.account.key."+ storage_account + ".blob.core.windows.net":storage_access_key})  

  display(dbutils.fs.mounts())

# COMMAND ----------

# DBTITLE 1,Create MLflow Experiment
#Check if experiment exists
if __name__ == "__main__":
  if not any(experiment.name ==experiment_name for experiment in mlflowclient.list_experiments()):
    experiment_id= mlflowclient.create_experiment(experiment_name, artifact_location="dbfs:"+ artifact_mountpoint +"/" + artifacts_folder)
    mlflow.set_experiment(experiment_name)
  else:
    experiment_id =mlflowclient.get_experiment_by_name(experiment_name).experiment_id
    mlflow.set_experiment(experiment_name)
    
  print("Experiment: {} Experiment ID: {}".format(experiment_name, experiment_id))

# COMMAND ----------

# DBTITLE 1,Train the model with MLflow Tracking
# if __name__ == "__main__":
  import numpy as np
  import os
  import shutil
  import datetime

  model_name = "AdultCensus.mml"
  model_dbfs = os.path.join("/dbfs", model_name)

  # Regularization Rates - 
  regs = [0.000001,0.00001,0.0001, 0.001, 0.01, 0.1]

  # try a bunch of regularization rate in a Logistic Regression model
  for reg in regs:
      print("Regularization rate: {}".format(reg))
      # create a bunch of child runs
      run_name="reg"+str(reg)
      with mlflow.start_run(experiment_id=experiment_id,run_name=run_name,nested=True) as run:
        
          # create a new Logistic Regression model.
          lr = LogisticRegression(regParam=reg)
        
          # put together the pipeline
          pipe = Pipeline(stages=[*si_xvars, *ohe_xvars, si_label, assembler, lr])

          # train the model
          model_p = pipe.fit(train)
        
          # make prediction
          pred = model_p.transform(test)
        
          # evaluate. note only 2 metrics are supported out of the box by Spark ML.
          bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
          au_roc = bce.setMetricName('areaUnderROC').evaluate(pred)
          au_prc = bce.setMetricName('areaUnderPR').evaluate(pred)

          print("Area under ROC: {}".format(au_roc))
          print("Area Under PR: {}".format(au_prc))
      
          # save model
          model_p.write().overwrite().save(model_name)
        
          # upload the serialized model into run history record
      
          mdl, ext = model_name.split(".")
          #mdl = mdl + "_" + experiment_id + "_" + run_id
          model_zip = mdl + ".zip"
          model_file =shutil.make_archive(mdl, 'zip', model_dbfs)
          print("model_file={}".format(model_file))
        
         #Mlflow Track Experiment
          print("run: {}".format(run.info))
          run_id=run.info.run_id
        
          # log reg, au_roc, au_prc and feature names in run history
          mlflow.log_param("feature_list", train.columns)
          mlflow.log_param("Regularization rate", reg)
          for traincol in train.columns:
            feature_name="feature" + str(train.columns.index(traincol))
            mlflow.log_param(feature_name, traincol)
        
          mlflow.log_metric("areaUnderROC", au_roc)
          mlflow.log_metric("areaUnderPR", au_prc)
        
          mlflow.set_tag("data",datafile)
          mlflow.set_tag("training data",train_data_path)
          mlflow.set_tag("test data",test_data_path)
          mlflow.set_tag("model_file",model_zip)
        
          mlflow.log_artifact(model_file)
          #mlflow.log_artifact(train_data_path_dbfs)
          #mlflow.log_artifact(test_data_path_dbfs)
          #mlflow.log_artifact(datafile_dbfs)       
        
     
          # now delete the serialized model from local folder since it is already uploaded to run history 
          shutil.rmtree(model_dbfs)
          os.remove(model_zip)
        
  # Declare run completed
  mlflow.end_run(status='FINISHED')