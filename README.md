# Disaster Response Pipeline Project

##  1. Business Understanding:
This project sets out to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. The project produces a machine learning pipeline to categorize these events to help match the messages with an appropriate disaster relief agency.The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.


##  2. Data Understanding:
The data comes from Figure Eight: https://www.figure-eight.com/

## 3. Data Preparation:
Two comma delimited text file (csv) are used in the project: messages and categories datasets. In the data preparation stage, the messages and categories datasets are merged using the common ID. The merged dataset is then cleaned through the following steps:
 - Split the values in the categories column on the ; character so that each value becomes a separate column
 - Use the first row of categories dataframe to create column names for the categories data
 - Rename columns of categories with new column names
 - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0)
 - Convert the string to a numeric value
 - Drop the categories column from the df dataframe
 - Concatenate df and categories data frames
 - Drop the duplicates
 - Store the cleaned dataset in a SQLite database

## 4. Modeling:
A machine learning pipeline is created in train_classifier.py, which performs the following tasks:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

## 5. Evaluation:
The final machine learning model chosen for production uses a Decision Tree model, which produces an accuracy rate of 94.6% on the test dataset. This model performed better compared to a random forest classifier, which produced an accuracy rate of 94.1% on the test dataset.

## 6. Structure of the Project

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model

- README.md

## 7. Instructions to run the app
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

**Author
Umrbek Allakulov**
