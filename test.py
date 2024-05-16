import re
import pandas as pd
import numpy as np
import json
import random
import pred as pr


if __name__ == "__main__":
    filename = "clean_dataset.csv"
    df = pd.read_csv(filename)
    X = df.drop("Label", axis=1)
    y = df["Label"]
    X_train, X_test, y_train, y_test = pr.train_test_split(X, y, test_size=0.2, random_state=70)
    nb = pr.NaiveBayes()
    for col in X_train.columns:
        Xfit = X[col, "Q1", "Q10"]
    nb.fit(Xfit, y_train)
    train_acc = pr.accuracy_score(nb.predict(Xfit), y_train)
    test_acc = pr.accuracy_score(nb.predict(X_test), y_test)
    print(f"{type(nb).__name__} train accuracy: {train_acc:.2f}")
    print(f"{type(nb).__name__} test accuracy: {test_acc:.2f}")

    #pr.predict_all(filename)

    filename = "clean_dataset.csv"
    m = predict_all(filename)
    print(m)
