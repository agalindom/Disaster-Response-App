# Disaster Response Pipeline Project

## Overview
The objective of this project was to build a web app that correctly classifies messages that are written in a disaster-like situation. The application was build on a dataset of already labeled tweets over 36 possible categories, provided by figure eight. The idea behind this app is for the different organizations in charge of handling these type of situations, to reduce their aid-reaction time by knowing in real time fashion, the type or category a message is.

## Description
The project was made in three different parts:

1. ETL pipeline:
    * Usage:
      - python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2. Machine learning model:
    * Usage:
      - python models/train_classifier.py data/DisasterResponse.db models/Random_Forest_cv.pkl

3. Flask app:
    * Usage:
      - python app/run.py

## Files

* data folder
  - init script
  - process_data.py: Code containing steps in building the ETL pipeline
  - disaster_categories.csv: Disaster categories of a particular message
  - disaster_messages.csv: Disaster messages
  - DisasterResponse.db: SQLite database where the new DataFrame was loaded

* models folder
  - init script
  - train_classifier: Code containing the NLP classification model

* app folder
  - init script
  - run.py: code containing the instructions to run the web app and plot visualization
  - templates folder: go.html and master.html, html code for displaying the web app.

## Libraries
If conda install the only libraries that need installation are Plotly and nltk.
