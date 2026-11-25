"""Analyze Spotify tracks and train models to predict track popularity."""

from pathlib import Path
import shutil
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "songs_normalize.csv"
RESULTS_DIR = ROOT / "project_results"
TARGET = "popularity"
NUMERICAL_FEATURES = [
    "duration_ms",
    "year",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
CATEGORICAL_FEATURES = ["artist", "song", "explicit", "key", "mode", "genre"]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURES + [TARGET]
POPULARITY_THRESHOLD = 50


def save_spotify_charts(data: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(data[TARGET], bins=30, edgecolor="black")
    plt.xlabel("Spotify Popularity (0-100)")
    plt.ylabel("Number of Tracks")
    plt.title("Distribution of Spotify Track Popularity")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "popularity_distribution.png")
    plt.close()

    genre_popularity = (
        data.groupby("genre")[TARGET].mean().sort_values(ascending=False)
    )
    ax = genre_popularity.plot(kind="bar", figsize=(12, 5))
    ax.set_xlabel("Genre")
    ax.set_ylabel("Average Popularity")
    ax.set_title("Average Track Popularity by Genre")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "popularity_by_genre.png")
    plt.close()

    year_popularity = (
        data.groupby("year")[TARGET].mean().sort_index()
    )
    plt.figure(figsize=(12, 5))
    plt.plot(year_popularity.index, year_popularity.values, marker="o")
    plt.xlabel("Release Year")
    plt.ylabel("Average Popularity")
    plt.title("Average Spotify Track Popularity by Year")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "popularity_by_year.png")
    plt.close()

    print("\nAverage popularity by genre:")
    print(genre_popularity.to_string())


def build_preprocessor() -> ColumnTransformer:
    numerical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def evaluate_regression(
    name: str, actual: pd.Series, predicted: np.ndarray
) -> dict[str, float | str]:
    mse = mean_squared_error(actual, predicted)
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2 Score": r2_score(actual, predicted),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Put songs_normalize.csv beside song.py."
        )

    df = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    print("Spotify dataset loaded successfully!")
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))

    df_clean = df.drop_duplicates().copy()
    for column in NUMERICAL_FEATURES + [TARGET]:
        df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")

    df_clean = df_clean.dropna(subset=[TARGET])
    df_clean = df_clean[df_clean[TARGET].between(0, 100)].copy()
    invalid_years = ~df_clean["year"].between(1950, 2035)
    df_clean.loc[invalid_years, "year"] = np.nan
    if df_clean.empty:
        raise ValueError("No usable Spotify tracks remain after cleaning.")

    print("\nOriginal dataset shape:", df.shape)
    print("Cleaned dataset shape:", df_clean.shape)
    df_clean.to_csv(ROOT / "songs_normalize_cleaned.csv", index=False)
    print("Cleaned dataset saved to songs_normalize_cleaned.csv")
    save_spotify_charts(df_clean)

    X = df_clean[FEATURES].copy()
    y = df_clean[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\nTraining shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    preprocessor = build_preprocessor()
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=42, min_samples_leaf=2
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    fitted_models = {}
    test_results = []
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        test_results.append(
            evaluate_regression(name, y_test, pipeline.predict(X_test))
        )
        print(f"{name} training completed!")

    baseline_results = pd.DataFrame(test_results).sort_values(
        by="MAE"
    ).reset_index(drop=True)
    baseline_results.to_csv(
        RESULTS_DIR / "baseline_model_comparison.csv", index=False
    )
    print("\nBaseline test-set results:")
    print(baseline_results.to_string(index=False))

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "MAE": "neg_mean_absolute_error",
                "MSE": "neg_mean_squared_error",
                "R2": "r2",
            },
            n_jobs=1,
        )
        mae_scores = -scores["test_MAE"]
        mse_scores = -scores["test_MSE"]
        cv_results.append(
            {
                "Model": name,
                "Mean CV MAE": mae_scores.mean(),
                "Std CV MAE": mae_scores.std(ddof=1),
                "Mean CV MSE": mse_scores.mean(),
                "Mean CV RMSE": np.sqrt(mse_scores).mean(),
                "Mean CV R2": scores["test_R2"].mean(),
            }
        )
        print(f"{name} cross-validation completed!")

    cv_results_df = pd.DataFrame(cv_results).sort_values(
        by="Mean CV MAE"
    ).reset_index(drop=True)
    cv_results_df.to_csv(RESULTS_DIR / "cross_validation_results.csv", index=False)

    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(random_state=42)),
        ]
    )
    dt_grid = GridSearchCV(
        estimator=dt_pipeline,
        param_grid={
            "model__max_depth": [5, 10, 20, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    dt_grid.fit(X_train, y_train)
    print("\nBest Decision Tree parameters:", dt_grid.best_params_)

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=1)),
        ]
    )
    rf_grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [20, None],
            "model__min_samples_leaf": [1, 2],
        },
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    rf_grid.fit(X_train, y_train)
    print("Best Random Forest parameters:", rf_grid.best_params_)

    tuning_results_df = pd.DataFrame(
        [
            {"Model": "Decision Tree", "Best CV MAE": -dt_grid.best_score_},
            {"Model": "Random Forest", "Best CV MAE": -rf_grid.best_score_},
        ]
    )
    tuning_results_df.to_csv(
        RESULTS_DIR / "hyperparameter_tuning_results.csv", index=False
    )

    final_models = {
        "Linear Regression": fitted_models["Linear Regression"],
        "Tuned Decision Tree": dt_grid.best_estimator_,
        "Tuned Random Forest": rf_grid.best_estimator_,
    }
    predictions = {
        name: model.predict(X_test) for name, model in final_models.items()
    }
    final_results = pd.DataFrame(
        [
            evaluate_regression(name, y_test, prediction)
            for name, prediction in predictions.items()
        ]
    )
    final_results.to_csv(
        RESULTS_DIR / "final_model_comparison.csv", index=False
    )
    print("\nFinal model test-set results:")
    print(final_results.to_string(index=False))

    actual_categories = np.where(
        y_test < POPULARITY_THRESHOLD, "Lower Popularity", "Higher Popularity"
    )
    for name, prediction in predictions.items():
        predicted_categories = np.where(
            prediction < POPULARITY_THRESHOLD,
            "Lower Popularity",
            "Higher Popularity",
        )
        cm = confusion_matrix(
            actual_categories,
            predicted_categories,
            labels=["Lower Popularity", "Higher Popularity"],
        )
        safe_name = name.lower().replace(" ", "_")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Lower Popularity", "Actual Higher Popularity"],
            columns=["Predicted Lower Popularity", "Predicted Higher Popularity"],
        )
        cm_df.to_csv(RESULTS_DIR / f"{safe_name}_confusion_matrix.csv")

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Lower Popularity", "Higher Popularity"],
            yticklabels=["Lower Popularity", "Higher Popularity"],
        )
        plt.xlabel("Predicted Category")
        plt.ylabel("Actual Category")
        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{safe_name}_confusion_matrix.png")
        plt.close()
        print(f"\n{name} classification report:")
        print(
            classification_report(
                actual_categories,
                predicted_categories,
                zero_division=0,
            )
        )

    plt.figure(figsize=(10, 5))
    plt.bar(final_results["Model"], final_results["MAE"])
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Mean Absolute Error")
    plt.title("Final Spotify Popularity Model Comparison - MAE")
    plt.xticks(rotation=15)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_model_comparison.png")
    plt.close()

    # Reuse baseline cross-validation and grid-search scores to select a model.
    selection_scores = {
        "Linear Regression": float(
            cv_results_df.loc[
                cv_results_df["Model"] == "Linear Regression", "Mean CV MAE"
            ].iloc[0]
        ),
        "Tuned Decision Tree": -float(dt_grid.best_score_),
        "Tuned Random Forest": -float(rf_grid.best_score_),
    }
    cv_selection_df = pd.DataFrame(
        [
            {"Model": name, "Mean CV MAE": score}
            for name, score in selection_scores.items()
        ]
    ).sort_values("Mean CV MAE").reset_index(drop=True)
    cv_selection_df.to_csv(
        RESULTS_DIR / "final_cross_validation_results.csv", index=False
    )
    best_model_name = cv_selection_df.loc[0, "Model"]
    joblib.dump(
        final_models[best_model_name],
        RESULTS_DIR / "best_spotify_popularity_model.pkl",
    )
    print("\nBest selected model:", best_model_name)
    print("Saved trained model and evaluation results in project_results.")

    archive_path = shutil.make_archive(
        str(ROOT / "Spotify_Project_Results"),
        "zip",
        root_dir=RESULTS_DIR,
    )
    print("Results archive:", archive_path)


if __name__ == "__main__":
    main()