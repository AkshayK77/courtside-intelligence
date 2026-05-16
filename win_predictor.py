"""
Step 3 — Train win prediction models on historical team data.
Compares LinearRegression, RandomForest, GradientBoosting via 5-fold CV.
Saves: win_model.pkl, feature_importance.csv
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = [
    'total_vorp',
    'top_player_vorp',
    'vorp_concentration',
    'avg_age',
    'avg_bpm',
]
TARGET = 'wins'


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv('training_data.csv')
    print(f"Loaded training_data.csv — {len(df)} rows, seasons: {sorted(df['season'].unique())}")

    # Drop rows with missing features or target
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"After dropping NaNs: {len(df)} rows")

    if len(df) < 10:
        print("WARNING: very few training rows — model accuracy will be low")

    X = df[FEATURES]
    y = df[TARGET]
    return df, X, y


def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series, cv: KFold) -> dict:
    r2_scores  = cross_val_score(model, X, y, cv=cv, scoring='r2')
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    return {
        'name':    name,
        'r2_mean': r2_scores.mean(),
        'r2_std':  r2_scores.std(),
        'mae_mean': mae_scores.mean(),
        'mae_std':  mae_scores.std(),
        'model':   model,
    }


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract normalised feature importances for any supported model type."""
    # Unwrap Pipeline if needed
    inner = model
    if hasattr(model, 'steps'):
        inner = model.steps[-1][1]

    if hasattr(inner, 'feature_importances_'):
        importances = inner.feature_importances_
    elif hasattr(inner, 'coef_'):
        importances = np.abs(inner.coef_)
        importances = importances / importances.sum()
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    df = pd.DataFrame({
        'feature':    feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    return df


def main():
    df, X, y = load_data()

    models = {
        'Linear Regression': Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())]),
        'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 62)
    print(f"{'Model':<25} {'R²':>8} {'±':>6}  {'MAE':>8} {'±':>6}")
    print("=" * 62)

    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X, y, cv)
        results.append(res)
        print(f"{name:<25} {res['r2_mean']:>8.4f} {res['r2_std']:>6.4f}  "
              f"{res['mae_mean']:>8.2f} {res['mae_std']:>6.2f}")

    print("=" * 62)

    # Pick the best model by mean R²
    best = max(results, key=lambda r: r['r2_mean'])
    print(f"\nBest model: {best['name']}  (R²={best['r2_mean']:.4f}, MAE={best['mae_mean']:.2f})")

    # Refit on full dataset
    best_model = best['model']
    best_model.fit(X, y)

    # Feature importance
    fi = get_feature_importance(best_model, FEATURES)
    print("\nFeature importances:")
    for _, row in fi.iterrows():
        bar = "█" * int(row['importance'] * 40)
        print(f"  {row['feature']:<25} {row['importance']:.4f}  {bar}")

    # Validation table — most recent season in the data
    latest_season = df['season'].max()
    latest = df[df['season'] == latest_season].copy()
    latest['predicted_wins'] = best_model.predict(latest[FEATURES]).round(1)
    latest['error'] = (latest['predicted_wins'] - latest[TARGET]).round(1)

    print(f"\nValidation table — season {latest_season}:")
    print(f"{'Team':<6} {'Actual':>7} {'Predicted':>10} {'Error':>7}")
    print("-" * 35)
    for _, row in latest.sort_values('error', key=abs).iterrows():
        team_col = 'team_abbr' if 'team_abbr' in latest.columns else 'team'
        print(f"{str(row.get(team_col, '?')):<6} {int(row[TARGET]):>7} "
              f"{row['predicted_wins']:>10.1f} {row['error']:>+7.1f}")

    # Save artifacts
    joblib.dump(best_model, 'win_model.pkl')
    fi.to_csv('feature_importance.csv', index=False)
    print("\nSaved → win_model.pkl, feature_importance.csv")

    # Also emit a small metadata file for the dashboard
    meta = {
        'best_model': best['name'],
        'r2':         round(best['r2_mean'], 4),
        'mae':        round(best['mae_mean'], 2),
        'top_feature': fi.iloc[0]['feature'],
    }
    pd.Series(meta).to_json('model_meta.json')
    print("Saved → model_meta.json")


if __name__ == '__main__':
    main()
