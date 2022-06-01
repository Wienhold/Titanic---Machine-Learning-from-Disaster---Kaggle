# Machine Learning Engineer for MS Azure - Capstone Project
# Titanic - Machine Learning from Disaster (Kaggle Competition)

After gaining some experience in working with Azure ML during my Udacity nanodegree I now want to finish the course with my capstone project.
As I always love a challenge, I chose to work on the Titanic Kaggle competition project.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
The dataset used in this project was downloaded from [Kaggle](https://www.kaggle.com/competitions/titanic/overview), where I also registered for the competition and later can submit my predictions. We are provided a [train](https://www.kaggle.com/competitions/titanic/data?select=train.csv) and a [test](https://www.kaggle.com/competitions/titanic/data?select=test.csv) dataset with 891 and 418 records, respectively.
We are given the following 11 parameters:
![image](https://user-images.githubusercontent.com/98894580/171413959-3d7347e6-3a4e-42f6-a359-697bc56b30ff.png)


### Feature Engineering
Descriptive statistics for the combined train+test dataset were received from Minitab statistical software. 
![image](https://user-images.githubusercontent.com/98894580/171394293-547806cf-0809-4a83-b508-0528e1dd18bb.png)

Here we can see that we have two incomplete feature columns: 418 missing values for age and 1 missing value for fare.
The later should be less relevant for our predictions so we will just fill it with the mean from the rest of the dataset (33.2955).
But the missing age values pose a relevant portion of all records in the dataset, so a more elaborate solution should be developed.
Using  Minitab's Predictive Analytics Module, a CART regression analysis was performed with Pclass, SibSp and Parch (ticket class, N of parents & spouses, N of parents & children) as categorical variables, aiming for least squared error and using a 10-fold cross validation.
As we see in the tree diagram, six terminal nodes were found.
![image](https://user-images.githubusercontent.com/98894580/171398053-62786b3e-5159-4df3-b9cb-a0cf9901a97a.png)
![image](https://user-images.githubusercontent.com/98894580/171398164-4ff00769-6c8a-45a1-9a24-ea1600c988d7.png)

It is obvious when we look at the residual plots that there is still a lot of deviation, which potentially could be decreased by further optimization. But for now, these values are sufficient to us and will be used for model development.
![image](https://user-images.githubusercontent.com/98894580/171398483-5b75645f-34fc-40f6-ae8e-cbe3c9f3b1b9.png)


### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

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
