# SBD Powerlifting Predictor

A machine learning project that predicts a lifter's competition squat, bench press, and deadlift totals from their age, bodyweight, and sex, wrapped in an interactive Streamlit interface. This is the third iteration of this project — the previous two versions were data analysis exercises rather than usable tools, and the main goal this time around was to turn that analysis into something an actual person could open and interact with.

## Motivation

I've been training in powerlifting (squat, bench, deadlift — "SBD") for a while, and I was curious whether a simple model could estimate someone's expected competition numbers just from a few basic stats. Behind that, this project was also a way to practice the full lifecycle of a small ML project,
- Cleaning a messy real-world dataset
- Iterating on features and models 
- Evaluating the models accuracy
and finally packaging the result into something interactive rather than leaving it as a notebook full of print statements.

I chose this as my project idea because it helps me practice and demonstrate the skills that I've practiced, while also being based on a personal passion that I am interested in.

## Dataset

The data comes from the Kaggle dataset [`us-powerlifting-competition-data-2015now`](https://www.kaggle.com/datasets/brianmcabee/us-powerlifting-competition-data-2015now), which contains results from USA powerlifting competitions since 2015. For this project, the data is filtered down to the USPA federation and cleaned to keep only complete, valid entries:

- Rows missing `Age`, `BodyweightKg`, `Sex`, or any of the three competition lifts are dropped.
- Non-ranked entries (DQs, no-shows, etc.) are excluded by requiring a numeric `Place`.
- Implausible lift-to-bodyweight ratios (e.g. a bench press more than double someone's bodyweight) are filtered out as likely data errors, with separate thresholds for men and women based on what's realistically achievable.

## Approach

### Exploratory analysis

Before modeling, I looked at the relationships between bodyweight, age, and each lift using scatter plots and histograms, split by sex. This confirmed the expected patterns — heavier lifters generally lift more in absolute terms, lifting peaks in the late 20s to mid-30s and tapers off with age, and the male and female distributions are distinct enough that sex needed to be a model input rather than something to normalize away.

### Modeling iterations

The first pass used a straightforward linear regression on `Age`, `BodyweightKg`, and `Sex` to predict each lift directly. It worked, but the residuals suggested the relationship between bodyweight and strength isn't perfectly linear — strength tends to increase with bodyweight at a diminishing rate.

To address this, I ran a series of feature engineering experiments, testing combinations of `log(BodyweightKg)`, `sqrt(BodyweightKg)`, `Age²`, and `Age³` against both linear regression and gradient boosting, evaluated with r², MAE, and RMSE.

The configuration that generalized best was:
- **Target:** `log(lift)` rather than the raw lift value, which stabilizes the variance across lighter and heavier lifters.
- **Features:** `Age`, `BodyweightKg`, `log(BodyweightKg)`, `Age²`, and one-hot encoded `Sex`.
- **Model:** Linear Regression, inside a `scikit-learn` `Pipeline` with a `ColumnTransformer` handling standard scaling and one-hot encoding.

Predicting the log of the lift and transforming back (via exponentiation) consistently outperformed predicting the raw kilograms directly, and kept the model honest about the fact that a 10 kg error means something very different for a 50 kg lifter than a 150 kg lifter.

### Results

Evaluated on a held-out 20% test split:

| Lift     | r²    | MAE (kg) | RMSE (kg) |
|----------|-------|----------|-----------|
| Squat    | 0.703 | 24.8     | 32.7      |
| Bench    | 0.790 | 17.0     | 22.6      |
| Deadlift | 0.716 | 25.1     | 32.5      |

Bench press is the most predictable of the three, which makes sense — it has the lowest technical variance and is less affected by leverages than squat or deadlift. An r² in the 0.70–0.79 range also makes sense: age, bodyweight, and sex clearly explain a large share of lifting strength, but individual variation in training history, technique, and genetics accounts for the rest, which a three-feature model can't capture.

## The App

The Streamlit app (`app.py`) retrains and exposes this final model through three tabs:

- **Predict** — enter an age, bodyweight, and sex to get an estimated squat, bench, deadlift, and total.
- **Explore the Data** — the same bodyweight/age scatter plots from the exploratory analysis, viewable interactively.
- **Model Performance** — the r²/MAE/RMSE table above, computed live from the trained model.

## Tech Stack

- **Language:** Python
- **Modeling:** scikit-learn (Pipeline, ColumnTransformer, LinearRegression)
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Data source:** kagglehub
- **Interface:** Streamlit
- **Prototyping:** Jupyter Notebook
- **Version control:** git / GitHub

## Project Structure

```
.
├── app.py              # Streamlit app
├── notebook/
│   └── main.ipynb      # exploratory analysis and model iteration
├── data/                # (populated at runtime via kagglehub cache)
├── models/
└── README.md
```

## Running Locally

1. Clone the repository and set up a virtual environment.
2. Install the dependencies:
   ```
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn kagglehub
   ```
3. Run the app:
   ```
   python -m streamlit run app.py
   ```

The first run downloads and caches the dataset via `kagglehub`; subsequent runs reuse the cached copy and the cached model.
