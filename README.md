Link to the github repository: https://github.com/aamster/udacity-disaster
Link to the github repository: https://github.com/aamster/udacity-disaster
Link to the github repository: https://github.com/aamster/udacity-disaster
Link to the github repository: https://github.com/aamster/udacity-disaster
Link to the github repository: https://github.com/aamster/udacity-disaster
Link to the github repository: https://github.com/aamster/udacity-disaster
Link to the github repository: https://github.com/aamster/udacity-disaster
Link to the github repository: https://github.com/aamster/udacity-disaster

# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Files:

- data/process_data.py: Loads data from files, cleans data, and loads into sqllite db
- models/train_classifier: Loads data from sqllite db, trains pipeline, evaluates model, and saves model to pickle file
- app/run.py: Launches web app which takes text as user input, and outputs classifications. Shows a couple visuals to
    better understand the data.
- notebooks/*: Exploratory notebooks
