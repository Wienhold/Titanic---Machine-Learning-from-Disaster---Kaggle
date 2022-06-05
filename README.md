# Machine Learning Engineer for MS Azure - Capstone Project
# Titanic - Machine Learning from Disaster (Kaggle Competition)

After gaining some experience in working with Azure ML during my Udacity nanodegree I now want to finish the course with my capstone project.
As I always love a challenge, I chose to work on the Titanic Kaggle competition project.

## Dataset

### Overview
The dataset used in this project was downloaded from [Kaggle](https://www.kaggle.com/competitions/titanic/overview), where I also registered for the competition and later can submit my predictions. We are provided a [train](https://www.kaggle.com/competitions/titanic/data?select=train.csv) and a [test](https://www.kaggle.com/competitions/titanic/data?select=test.csv) dataset with 891 and 418 records, respectively.
We are given the following 12 columns, here some examplary data:
![image](https://user-images.githubusercontent.com/98894580/171413959-3d7347e6-3a4e-42f6-a359-697bc56b30ff.png)

As we can already see from the first few records, there are some missing values. Our first task will be to handle those.

### Feature Completion
Descriptive statistics for the combined train+test dataset were received from Minitab statistical software. 
![image](https://user-images.githubusercontent.com/98894580/171394293-547806cf-0809-4a83-b508-0528e1dd18bb.png)

Here we can see that we have three incomplete feature columns: 418 missing values for `Age`, two missing values for `Embarked` and 1 missing value for `Fare`.
The later should be less relevant for our predictions, so we will just fill `Embarked` with a random selection from the three options and `Fare` with the mean from the rest of the dataset (33.2955).
But the missing age values pose a relevant portion of all records in the dataset, so a more elaborate solution should be developed.
Using  Minitab's Predictive Analytics Module, a CART regression analysis was performed with Pclass, SibSp and Parch (ticket class, N of parents & spouses, N of parents & children) as categorical variables, aiming for least squared error and using a 10-fold cross validation.
As we see in the tree diagram, six terminal nodes were found.
![image](https://user-images.githubusercontent.com/98894580/171398053-62786b3e-5159-4df3-b9cb-a0cf9901a97a.png)
![image](https://user-images.githubusercontent.com/98894580/171398164-4ff00769-6c8a-45a1-9a24-ea1600c988d7.png)

It is obvious when we look at the residual plots that there is still a lot of deviation, which potentially could be decreased by further optimization. But for now, these values are sufficient to us and will be used for model development.
![image](https://user-images.githubusercontent.com/98894580/171398483-5b75645f-34fc-40f6-ae8e-cbe3c9f3b1b9.png)


### Task
From the given features for each record we want to make a prediction if this person did survive the Titanic disaster or not. Therefore relevant features should be identified to help us with this task.
Used features:
- Pclass: integer, ticket class (1, 2, 3), proxy for wealth
- Sex: string, male or female
- Age: float, age in years
- Sibsp: integer, number of siblings and/or spouses aboard (brother, sister, stepbrother, stepsister, husband, wife)
- Parch: integer, number of parents and/or children aboard (mother, father, daughter, son, stepdaughter, stepson)
- Fare: float, amoung of money paid for the ticket
- Embarked: string, port of embarkation	(C = Cherbourg, Q = Queenstown, S = Southampton)

Discarded features:
- PassengerId: integer, internal key
- Name: string
- Ticket: string, ticket numbers as various formats
- Cabin: string, cabin number, many missing values

Target:
- Survived: binary, ground truth

So we will use 7 features for training and prediction.

### Access
The training dataset as well as the testing dataset are registered within MS Azure. Furthermore, respective filtered versions with only the relevant features specified above as well as with the target "Survived" for the training dataset were registered as well.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
`automl_settings = {"max_concurrent_iterations": 5,
"max_cores_per_iteration": -1,
"enable_dnn": True,
"enable_early_stopping": True,
"validation_size": 0.2,
"primary_metric" : 'accuracy',
"enable_voting_ensemble": False,
"enable_stack_ensemble": False }

automl_config = AutoMLConfig(compute_target=compute_target,
task = "classification",
training_data=dataset_filtered,
label_column_name="Survived",
path = project_folder,
featurization= 'auto',
debug_log = "automl_errors.log",
**automl_settings )`

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
