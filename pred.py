import re
import pandas as pd
import numpy as np
import json
import random



#file_name = "clean_dataset.csv"
random_state = 86

def word_count(X,y):
    labels = np.unique(y)
    word_counts = {label: {} for label in labels}
    word_counts["total"] = {}

    for label in labels:
        y_copy = y.copy()
        new_x = X[y_copy == label]
        label_data = new_x.copy()
        word_counts[label + "size"] = len(label_data)
         ## getq10 only 
        for text in label_data["Q10"]:
            if not isinstance(text, str):
                text = 'impossibledetrouvercemot'
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if word in word_counts[label] :
                    word_counts[label][word] += 1
                else:
                    word_counts[label][word] = 1

                if word in word_counts["total"] :
                    word_counts["total"][word] += 1
                else:
                    word_counts["total"][word] = 1
    return word_counts

def Probabilities_count(X,y):
    word_probs = {} #word gives label # gives intersection.
    y_copy = y.copy()
    labels = np.unique(y_copy)
    total_num = len(X)
    #dic_sizes = {}
    if 'Q10' in X.columns:
        word_counts = word_count(X,y_copy)   
        for label in labels:
            for word, count in word_counts[label].items():
                if word not in word_probs:
                    word_probs[word] = {}
                word_total =  word_counts["total"][word]
                if count/word_total > 0.5:

                    for lab in labels:
                        if word in word_counts[lab]:                       
                            word_probs[word][lab]= word_counts[lab][word]/ word_total
                        else :
                            word_probs[word][lab]= ( (word_counts["total"][word] - count + 1)* 20/80) / (2 * word_total * 20/80)
                else:
                    if word_probs[word] == {}:
                        word_probs.pop(word)
 
    return word_probs


        
        # take all the words in a given label and all the words 
    # if the word for a certain label probability is bigger then the probability of taking a line randomely and the word is in it.
   



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
        # create the dictionnary for words
        self.word_probs = Probabilities_count(X,y)
        self.classes = np.unique(y)
        self.class_probs = {}
        self.mean = {}
        self.var = {}
        columns_to_include = [col for col in X.columns if col != 'Q10']
        X_filtered = X[columns_to_include]  
        for c in self.classes:
            X_c = X_filtered[y == c]
            self.class_probs[c] = len(X_c) / len(X)
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            
        self.__save_model__()
    
    def predict(self, X):
        columns_to_include = [col for col in X.columns if col != 'Q10']
        X_filtered = X[columns_to_include]  
        Q10_data = X['Q10'] if 'Q10' in X.columns else None
        if isinstance(X_filtered, pd.DataFrame):
            X_new = X_filtered.values
        predictions = []
        #get q10 input verify if its self.word_probs for each word n it and multiply the probability by the liqulihood for each label under 
        for index, x in enumerate(X_new):
            posteriors = {}
            for c in self.classes:
                likelihood = 1
                for i in range(len(x)):
                    likelihood *= self.gaussian_likelihood(x[i], self.mean[c][i], self.var[c][i])
                if Q10_data is not None:
                    q10_text = Q10_data.iloc[index]  # Access the specific 'Q10' value for the current row
                    words = re.findall(r'\b\w+\b', str(q10_text).lower())  # Ensure q10_text is a string
                    for word in words:
                        if word in self.word_probs and c in self.word_probs[word]:
                            likelihood *= self.word_probs[word][c]
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
            f.write(f"Classes: {self.classes.tolist()}\n")
            for class_label, probability in self.class_probs.items():   
                f.write(f"{class_label}: {probability}\n")

            f.write("Mean:\n")
            for label, means in self.mean.items():  # Assuming self.mean is a dictionary
                means_str = ', '.join([str(m) for m in means])  # Convert each mean value to string and join with comma
                f.write(f"{label}: [{means_str}]\n")

            # Write the 'Variance' section
            f.write("Variance:\n")
            for label, variances in self.var.items():  # Assuming self.var is a dictionary
                variances_str = ', '.join([str(v) for v in variances])  # Convert each variance value to string and join with comma
                f.write(f"{label}: [{variances_str}]\n")

        with open("Bays.json", 'w') as f:
            json.dump(self.word_probs, f)

    def load_model(self):
        with open("Bays.json", 'r') as f:
            self.word_probs = json.load(f)
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

    if random_state is not None:
        random.seed(random_state)
    num_samples = len(X)
    num_test_samples = int(test_size * num_samples)

    indices = list(range(num_samples))
    random.shuffle(indices)


    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    # Split the data based on the indices
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test

def accuracy_score(y_true, y_pred):
    if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
    y_true = np.array(y_true)
    correct_predictions = 0
    
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            correct_predictions+=1

    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def clean_data(df):
    num_late_columns = ["Q7", "Q8", "Q9"]
    for col in num_late_columns:
        if col in df.columns:
            df[col] = df[col].apply(to_numeric).fillna(0)
           # df = replace_outliers_by_label(df, col, "Label")


    num_columns = ["Q1", "Q2", "Q3", "Q4"]
    for col in num_columns:
        if col in df.columns:
            df[col] = df[col].apply(get_number)

    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
        if "Q5" in df.columns:
            df[f"Q5_{cat}"] = df["Q5"].apply(lambda s: cat_in_s(s, cat))

    if "Q6" in df.columns:
        df["Q6"] = df["Q6"].apply(get_number_list_clean)
        rank_names = ["Skyscrapers", "Sport", "Art and Music", "Carnival", "Cuisine", "Economic"]
        for i, name in enumerate(rank_names):
            df[name] = df["Q6"].apply(lambda x: x[i])

    col_removal = ["Q5", "Q6", "id"]
    for col in col_removal:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)  #"Q10", 

    for col in (num_columns + num_late_columns + rank_names):
        df[col] = standardize_column(df[col])



def predict_all(filename):
    df = pd.read_csv(filename)
    clean_data(df)
    #y = df["Label"]
    #.drop("Label", axis=1)
    X = df
    nb = NaiveBayes()
    #nb.fit(X_train,y_train)
    nb.load_model()
    return nb.predict(X)