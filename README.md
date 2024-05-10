# Customer Churn Model

This is a customer churn prediction model built with Streamlit, based on the Kaggle Dataset - Telco Customer Churn. The goal is to explore the dataset from the perspective of a Telco business manager and understand which parameters can be adjusted to reduce customer churn.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your_username/your_repository.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

## Features

- **Data Exploration**: View and filter the Telco customer dataset.
- **Centroids**: Explore cluster centroids obtained from K-Means clustering.
- **Search**: Search for specific customers or centroids based on customer ID or cluster number.
- **Customized Use-case**: Customize user parameters to predict churn probability.
- **Churn Prediction**: Predict whether a customer will churn or not.
- **Visualization**: Visualize churn probability with pie charts.

## Files and Directories

- **app.py**: Main Streamlit application file.
- **Data.csv**: Dataset containing Telco customer information.
- **Data_centroids.csv**: CSV file containing cluster centroids data.
- **Searched_Records.csv**: CSV file to store searched records.
- **finalized_model.sav**: Pickle file containing the trained random forest model.
- **requirements.txt**: File containing the required Python libraries.
- **README.md**: Documentation file explaining the project.

## Libraries Used

- Streamlit
- NumPy
- Pandas
- Pickle
- Matplotlib
- scikit-learn
