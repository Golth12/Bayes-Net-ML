"""
This Python file provides some useful code for reading the training file
"clean_dataset.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


file_name = "clean_dataset.csv"
random_state = 42

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]

def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """

    n_list = get_number_list(s)
    n_list += [-1]*(6-len(n_list))
    return n_list

def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1

def find_area_at_rank(l, i):
    """Return the area at a certain rank in list `l`.

    Areas are indexed starting at 1 as ordered in the survey.

    If area is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

def replace_outliers_by_label(df, column_to_filter, label_column):
    """Replace outliers in a specific column within each label group with the median of the group."""
    labels = df[label_column].unique()
    
    for label in labels:
        label_data = df[df[label_column] == label]
        
        # Check if label_data is empty or if column_to_filter only has one unique value
        if label_data.empty or label_data[column_to_filter].nunique() == 1:
            print(f"No data to process or only one unique value for label: {label}")
            continue
        
        # Ensure the column is numeric and has no NaNs
        label_data[column_to_filter] = label_data[column_to_filter].apply(to_numeric).fillna(label_data[column_to_filter].median())
        
        Q1 = label_data[column_to_filter].quantile(0.25)
        Q3 = label_data[column_to_filter].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median_value = label_data[column_to_filter].median()
        
        df.loc[(df[label_column] == label) & ((df[column_to_filter] < lower_bound) | (df[column_to_filter] > upper_bound)), column_to_filter] = median_value
    
    return df

def standardize_column(column):
    mean = column.mean()
    std = column.std()
    # Avoid division by zero in case of a constant column
    if std != 0:
        column = (column - mean) / std
    return column


if __name__ == "__main__":
    df = pd.read_csv(file_name)

    # Apply numeric conversion and outlier replacement
    numeric_columns = ["Q7", "Q8", "Q9"]
    for col in numeric_columns:
        df[col] = df[col].apply(to_numeric).fillna(0)
        df = replace_outliers_by_label(df, col, "Label")

    # Convert ordinal features to numeric
    ordinal_columns = ["Q1", "Q2", "Q3", "Q4"]
    for col in ordinal_columns:
        df[col] = df[col].apply(get_number)

    # Create binary indicators for categorical features in Q5
    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
        df[f"Q5_{cat}"] = df["Q5"].apply(lambda s: cat_in_s(s, cat))

    # Handle rank data in Q6
    df["Q6"] = df["Q6"].apply(get_number_list_clean)
    rank_names = ["Skyscrapers", "Sport", "Art and Music", "Carnival", "Cuisine", "Economic"]
    for i, name in enumerate(rank_names):
        df[name] = df["Q6"].apply(lambda x: x[i])

    # Drop the original Q5 and Q6 columns, as well as any other unnecessary columns
    df.drop(["Q5", "Q6", "Q10", "id"], axis=1, inplace=True)

    # Standardize the continuous and ordinal features
    for col in (ordinal_columns + numeric_columns + rank_names):
        df[col] = standardize_column(df[col])

    # Prepare the data for training
    X = df.drop("Label", axis=1)
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)


random_forest_clf = RandomForestClassifier(random_state=random_state)
# these are the hyperparameters we are going to be choosing over
param_grid = {
    'n_estimators': [100, 200, 300, 400],  # trees in the forest
    'max_depth': [None, 5, 10, 20, 30],  # depth of the tree
    'min_samples_split': [2, 5, 10],  # min samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # min number of samples to be at a leaf node
}

# To find the best parameters
grid_search_rf = GridSearchCV(estimator=random_forest_clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit GridSearchCV
grid_search_rf.fit(X_train, y_train)

print(f"Best parameters: {grid_search_rf.best_params_}")
print(f"Best cross-validation score: {grid_search_rf.best_score_}")

# Use the best estimator to make predictions
best_estimator_rf = grid_search_rf.best_estimator_
y_pred_rf = best_estimator_rf.predict(X_test)

# Evaluate the classifier
print(f"Test accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))

# For max_depth = 10, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 300