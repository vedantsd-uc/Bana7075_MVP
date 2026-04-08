# Bana7075_MVP
Machine learning model for Bana 7075 Machine Learning course

Machine learning model trained to predict demand for rental bikes. Based off of dataset pulled from [Kagle Bike Rental](https://www.kaggle.com/competitions/bike-sharing-demand/data)

# Setup
Utilizing Python version 3.13

## Install Dependancies
Project dependancies are listed in the requirements.txt file. Install dependancies from file:
`pip install -r requirements.txt`

# Development Notes
## Update Dependancies
When new dependancies are added to the project, make sure to add them to the requirements file:
`pip freeze > requirements.txt`

# Model Experiments
MLFlow is the primary tool to manage experiment tracking and model versioning

Utilize the MLFlow ui by running the following command in terminal:
`mlflow ui`

Then navigate to the locally hosted webpage
(http://127.0.0.1:5000)

# Usage
Run full pipeline
`python main.py`

Run data quality tests
`python -m tests.data_quality.test_data_quality`

# Contributors
James Allen​
Lee Brodbeck-Moore
Rachael Rahe
Brett Toothman
Sagar Vedantam