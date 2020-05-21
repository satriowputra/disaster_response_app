
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)

## Installation <a name="installation"></a>

To run code provided here, you should have Flask and Plotly installed on your machine. There should be no installation required if you have Anaconda installed.

## Project Motivation<a name="motivation"></a>

By doing this project, I want to apply my skill in Data Engineering to create data ETL and in machine learning to build NLP pipelines. I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages. I also included a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the dataset.

## File Descriptions <a name="files"></a>

There are 2 python script provided here which are:
	1. ETL Pipeline Script, process_data.py. This will load and clean the datasets listed below and prepared the dataset to be used in machine learning algorithm. The script then creates SQLite database as output.	
	2. Machine Learning Pipeline Script, train_classifier.py. This script will load data, train NLP model, pick the best parameters (using gridsearch), print the results, and save it for you. You can directly use the model for you application.

In this project, dataset is provided by Figure-Eight. The dataset included are:
	1. message.csv
		In this file, there are 30000 messages extracted from twitter during several disaster such as an earthquake in Chile in 2010, floods in Pakistan in 2010, and other disasters.
	2. categories.csv
		This file contains 36 classification of messages inside dataset above.

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here].

If you want to run web app locally, run app.py and go to http://0.0.0.0 to see the app.

