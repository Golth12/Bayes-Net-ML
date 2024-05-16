import re
import pandas as pd
import numpy as np
import random
import csv

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

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probs = {}
        self.mean = {}
        self.var = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.class_probs[c] = len(X_c) / len(X)
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)

        self.__save_model__()
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                likelihood = 1
                for i in range(len(x)):
                    likelihood *= self.gaussian_likelihood(x[i], self.mean[c][i], self.var[c][i])
                if c in self.class_probs:  # Check if c is a key in class_probs
                    posterior = likelihood * self.class_probs[c]
                    posteriors[c] = posterior
                else:
                    print(f"Class {c} not found in class_probs")
            if posteriors:  # Ensure posteriors is not empty
                predictions.append(max(posteriors, key=posteriors.get))
        return predictions

    
    def gaussian_likelihood(self, x, mean, variance):
        return np.exp(-((x - mean) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)
    
    def __save_model__(self):
        model_file = 'naive_bayes_model.txt'

        with open(model_file, 'w') as f:
            f.write(f"Classes: {nb.classes.tolist()}\n")
            f.write(f"Class probabilities: {nb.class_probs.tolist()}\n")
            f.write("Mean:\n")
            for c, mean in nb.mean.items():
                f.write(f"{c}: {mean.tolist()}\n")
            f.write("Variance:\n")
            for c, var in nb.var.items():
                f.write(f"{c}: {var.tolist()}\n")

    def load_model(self):
        model_file = 'naive_bayes_model.txt'
        with open(model_file, 'r') as f:
            lines = f.readlines()
            classes = eval(lines[0].split(": ")[1])
            class_probs = {}
            mean = {}
            var = {}
            mean_started = False
            var_started = False
            for line in lines[1:]:
                if line.startswith("Mean:"):
                    mean_started = True
                    var_started = False
                    continue
                elif line.startswith("Variance:"):
                    mean_started = False
                    var_started = True
                    continue
                
                if mean_started:
                    class_name, mean_str = line.strip().split(": ")
                    mean[class_name] = np.array(eval(mean_str))
                elif var_started:
                    class_name, var_str = line.strip().split(": ")
                    var[class_name] = np.array(eval(var_str))
                else:
                    parts = line.strip().split(": ")
                    if len(parts) == 2:
                        class_name, prob_str = parts
                        class_probs[class_name] = float(prob_str)

        self.classes = classes
        self.class_probs = class_probs
        self.mean = mean
        self.var = var

def train_test_split(X, y, test_size=0.2, random_state=None):
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
    
    # Convert X and y to numpy arrays if they are pandas DataFrames
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Calculate number of samples for the test set
    num_test_samples = int(test_size * len(X))
    
    # Shuffle the indices
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # Split the indices into training and testing indices
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]
    
    # Split the data based on the indices
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    correct_predictions = 0
    
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            correct_predictions+=1

    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def clean_data(df):
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


if __name__ == "__main__":
    df = pd.read_csv(file_name)
    clean_data(df)

    X = df.drop("Label", axis=1)
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    nb = NaiveBayes()
    nb.fit(X_train,  y_train)
    nb.load_model()

    train_acc = accuracy_score(nb.predict(X_train), y_train)
    test_acc = accuracy_score(nb.predict(X_test), y_test)

    print(f"{type(nb).__name__} train accuracy: {train_acc:.2f}")
    print(f"{type(nb).__name__} test accuracy: {test_acc:.2f}")




