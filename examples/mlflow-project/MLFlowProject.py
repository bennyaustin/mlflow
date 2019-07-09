# Databricks notebook source
import mlflow
from mlflow.projects import *
from mlflow.tracking import *

mlflowclient= MlflowClient() 
experiment_name="/Shared/MLFlow/IncomePrediction"

# Disabling Azure Storage as MLflow can't recognize dbutils
# secret_scope="dbvault"
# storage_account="storageg2"
# storage_access_key= dbutils.secrets.get(secret_scope,"storageg2_key")
# artifact_mountpoint="/mnt/artifact"
# artifacts_folder="IncomePrediction"

# if not any(mount.mountPoint ==artifact_mountpoint for mount in dbutils.fs.mounts()):
#   dbutils.fs.mount( source = "wasbs://ml-artifacts@" + storage_account + ".blob.core.windows.net",
#           mount_point = artifact_mountpoint,
#           extra_configs = {"fs.azure.account.key."+ storage_account + ".blob.core.windows.net":storage_access_key})  

# # display(dbutils.fs.mounts())


#Check if experiment exists
if not any(experiment.name ==experiment_name for experiment in mlflowclient.list_experiments()):
  # experiment_id= mlflowclient.create_experiment(experiment_name, artifact_location="dbfs:"+ artifact_mountpoint +"/" + artifacts_folder)
  experiment_id= mlflowclient.create_experiment(experiment_name)
  mlflow.set_experiment(experiment_name)
else:
  experiment_id =mlflowclient.get_experiment_by_name(experiment_name).experiment_id
  mlflow.set_experiment(experiment_name)

print("Experiment: {} Experiment ID: {}".format(experiment_name, experiment_id))


# Cluster spec dictionary
cluster_spec ={"spark_version":"5.3.x-cpu-ml-scala2.11",
               "num_workers": 2,
               "node_type_id": "Standard_F4s",
              }
# type(cluster_spec_file)

mlflow.projects.run("https://github.com/bennyaustin/mlflow#examples/mlflow-project",entry_point="MLFlowTracking.py",use_conda=False,experiment_id=experiment_id,backend="databricks",backend_config=cluster_spec,synchronous =True)