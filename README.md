# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run. 

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The data is related with direct marketing campaigns of a banking institution.
The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.The data contained information about the job, marital status, if the customer has been previously contacted.
The dataset is saved as a csv file. It has information  on 32950 people. 
In this project we compared the accuracy of SKLEARN pipelina against an automl pipeline. 

The sklearn best model parameters were:
* run_id: 'HD_5a197d24-87f4-4442-81b0-5360fc2c6d19_0',
* hyperparameters: '{"--C": 0.01, "--max_iter": 100}',
*  accuracy: 0.9166919575113809

The autoMl best model parameter:
* accuracy: 0.9176024279210926
* algorithm:. Voting Ensemble


## Scikit-learn Pipeline
The SKLEARN pipeline steps are the following:
* Downloading the data provided in the tarin.py file
* clean data:
  * rows with missing values dropped (though as it happens, none were removed as the cleaned dataset still had 32950 rows, so there were no missing values)
  * the job, education and contact columns one-hot encoded
  * the marital, default, housing and loan columns encoded numerically with positive values being encoded as 1
  * the month and day_of_week columns being encoded as numerical values as per the dictionaries defined in the script
  * the poutcome column being encoded as a 1 for values of "success" and 0 otherwise
  * the y column being split out into a separate dataframe and encoded as 1 for the "yes" values, 0 otherwis
In the pipeline, the parameter to use in the model ( logistic regression) have been defined as following:

```
ps = RandomParameterSampling(
{'--C':choice(0.01,0.1,1,10,50,100,500,1000),
'--max_iter':choice(50,100,200,300)})
```
max_iter: the max number of iterations
C: regularization parameter
RandomParameterSampling is one of the choices available for the sampler. It is the faster and supports early termination of low-performance runs

**early stopping policy**
The policy early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run.
```
policy = BanditPolicy(evaluation_interval=1,slack_factor=0.1)
```
* The Bandit Policy Class defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation.

* slack_factor: The ratio used to calculate the allowed distance from the best performing experiment run.
* evaluation_interval: The frequency for applying the policy.

reference: https://learn.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py
## AutoML
```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=train_data,
    label_column_name='y',
    n_cross_validations=5)
```

* experiment_timeout_minutes: This is an exit criterion and is used to define how long, in minutes, the experiment should continue to run. 
* task: The type of task to run. Values can be 'classification', 'regression', or 'forecasting' depending on the type of automated ML problem to solve.
* primary_metric: The metric that Automated Machine Learning will optimize for model selection. Automated Machine Learning collects more metrics than it can optimize. You can use get_primary_metrics to get a list of valid metrics for your given task. For more information on how metrics are calculated, see
*n_cross_validations How many cross validations to perform when user validation data is not specified.
## Pipeline comparison
As mentioned before, the accuracy diffeence between the two models is trivial. In my opinion, to make an honest comparison, we should re run the experiment giving more time to the AutoMl and more run to the SKLEARN pipeline( to find the best values copuple).
AutoML is completely automatic, but you loose the sense of control on it, so you are not deciding what is the best for your model.
In my opinion, it woul be better always compare the results from these two pipelines to obtain the best results. 

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
* Give more time to autoML to run
* The autoMl alerted that the dataset is imbalanced. So, we'd need more data to balnace the dataset, or resampling the dataset to balance it. 
## Proof of cluster clean up
![image](https://github.com/AnnaDM87/Azure_project_1/assets/22540529/0d272d0d-ef7a-4642-9470-d86c1071500b)

