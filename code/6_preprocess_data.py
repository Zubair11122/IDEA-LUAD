#!/usr/bin/env python3
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
def load_and_preprocess(raw_path="mutations_variant_complete.tsv"):
    # Load data
    df = pd.read_csv(raw_path, sep='\t')

    # Convert target variable
    df['is_driver'] = df['is_driver'].astype(int)

    # Convert numeric columns
    numeric_cols = [
        'af', 'dp',
        'cadd.phred', 'cadd.score',
        'sift.score', 'sift.rankscore',
        'polyphen2.hdiv_rank', 'polyphen2.hvar_rank',
        'revel.score', 'revel.rankscore',
        'gerp.gerp_rs', 'gerp.gerp_rs_rank',
        'alphamissense.am_pathogenicity'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Feature selection
    cat_feats = [
        'variant_classification',
        'cancer_type',
        'dominant_signature',
        'so',
        'sift.prediction'
    ]

    candidate_num = [
                        'af', 'dp',
                        'cadd.phred',
                        'sift.score',
                        'revel.score',
                        'alphamissense.am_pathogenicity',
                        'gerp.gerp_rs'
                    ] + [c for c in df.columns if c.startswith("sbs")]

    X = df[cat_feats + candidate_num]
    y = df['is_driver']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Identify numeric features with at least some non-missing values
    num_feats = [f for f in candidate_num if X_train[f].notna().any()]
    dropped = set(candidate_num) - set(num_feats)
    if dropped:
        print(f"⚠️ Dropping all-missing numeric features: {dropped}")

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
    ])

    # Apply preprocessing
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Get feature names
    feature_names = preprocessor.get_feature_names_out()

    # Create DataFrames with proper column names
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)

    # Save preprocessor
    joblib.dump(preprocessor, "preprocessor.pkl")

    return X_train_df, X_test_df, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print(f"✅ Preprocessing complete. Shapes — train: {X_train.shape}, test: {X_test.shape}")