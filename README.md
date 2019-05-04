# Disaster Response Pipeline Project

## Overview
The objective of this project was to build a web app that correctly classifies messages that are written in a disaster-like situation. The application was build on a dataset of already labeled tweets over 36 possible categories, provided by figure eight. The idea behind this app is for the different organizations in charge of handling these type of situations, to reduce their aid-reaction time by knowing in real time fashion, the type or category a message is.

## Description
The project was made in three different parts:

1. ETL pipeline - located in /data:

  * Files:
    - init script
    - process_data.py: Code containing steps in building the ETL pipeline
    - disaster_categories.csv: Disaster categories of a particular message
    - disaster_messages.csv: Disaster messages
    - DisasterResponse.db: SQLite database where the new DataFrame was loaded

  * Steps:
    - Extract and merge disaster_categories.csv with disaster_messages.csv

    - Preprocess the new merged DataFrame

    - Load it into an SQLite database

  * Usage:

    - To run ETL pipeline that cleans data and stores in database:
      python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
