### Data-Centric AI Challenge: Forest Cover Optimization
Enhancing Machine Learning performance through advanced Feature Engineering on a fixed Random Forest model.

## Project Context
This project was developed for the Data-Centric AI Challenge. The objective is to improve the F1-Score of a forest cover classification system. In this specific industrial context, the predictive model (a Random Forest) is fixed and integrated into a critical C++ system; its hyperparameters cannot be modified.

Consequently, the entire optimization strategy relies on Data Engineering: cleaning, transforming, and enriching the input features to maximize the model's predictive accuracy.

## Technical Implementations
The preprocessing pipeline implemented in preprocessing_MUKAM_Lavoisier.py applies several advanced feature engineering techniques to a dataset of 581,012 observations:

Numerical Safety & Pipeline Robustness: Automated handling of missing values (NaN) and infinite values to ensure execution stability.

Geospatial Engineering:

Trigonometric Aspect Transformation: Conversion of the "Aspect" (azimuth) into Sine and Cosine components to preserve the circular nature of compass directions.

Euclidean Metrics: Calculation of direct Euclidean distances to hydrology, providing a more relevant spatial metric than raw horizontal and vertical components.

Topographical Interactions: Synthesis of new features combining Elevation and Slope to capture complex environmental relationships.

Categorical Optimization: Efficiently collapsing high-cardinality One-Hot encoded features (40 Soil Types and 4 Wilderness Areas) into dense categorical codes to reduce dimensionality while preserving information.

Lighting Synthesis: Analysis of Hillshade data across different times of day to extract mean solar intensity and range.

## Technical Stack
Language: Python 

Core Libraries: Pandas, NumPy 

Evaluation Baseline: Random Forest Classifier (50 estimators, max depth of 6).

## Usage
The project is structured around the mandatory transform_data function:

Python
from preprocessing_MUKAM_Lavoisier import transform_data
import pandas as pd

# Load the raw dataset
df_raw = pd.read_csv("covtype.csv")

# Apply the advanced preprocessing pipeline
X_transformed = transform_data(df_raw)
