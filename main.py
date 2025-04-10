import pandas as pd
import numpy as np

import matplotlib.pylab as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# dataset= Data frame , a common variable to define a variable in pandas
dataset = pd.read_csv("framingham.csv")
# rint(dataset.head())

# null function of pandas tells how many of out data columns or row is null and add them to tell the total of colums missing
# print(dataset.isnull().sum())
# .to_frame().T the same as above but in column way
# print(dataset.isnull().sum().to_frame().T)
# print(dataset.shape) #To let know how many row and columns are there

# As we have seen theres a lot of null columns in our dataset
# We will fill those
# First we will see the which columns have binary value, True o r false or 0,1
binary_col = ["male", "currentSmoker", "prevalentStroke", "prevalentHyp", "diabetes"]

for col in binary_col:
    mode_value = dataset[col].mode()[0]
    dataset[col] = dataset[col].fillna(mode_value)  # ← this assignment is crucial!

# print(missing_values)
# Similarly we will the the data col for numerical value not binary
# Even there the characters or alphabets in dataset it will be converted coz we are using numpy
# The alphbatical things should be only True and False i meant

numerical_col = ["cigsPerDay", "BPMeds", "totChol", "BMI", "heartRate", "glucose"]
# Tip do not include the target column in binary fill or numerical fill
for col in numerical_col:
    median_value = dataset[col].median()
    dataset[col] = dataset[col].fillna(median_value)
missing_values = dataset.isnull().sum().to_frame().T
# print(missing_values)
# Now the whole data set is filled , we are good to go ahead:

# print(dataset['TenYearCHD'].value_counts()) #Now we will see by callue value counts ofpd , that there a imbalce in out target column
# Balancing out Target Column

# AS SEEN THAT 1 IS IN MINORITY AND 0S ARE IN MAJORITY
majority = dataset[dataset["TenYearCHD"] == 0]
minority = dataset[dataset["TenYearCHD"] == 1]

minority_unsampling = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42,  # $@ is kinda standard then None i guess
    stratify=None,
)
balanced_data = pd.concat([majority, minority_unsampling])

# print(balanced_data['TenYearCHD'].value_counts()) #Now we will se that the 1s and 0s are in same qty

# Now Training and Testing Split of dataset
X = balanced_data.drop(columns=["education", "TenYearCHD"]).values
y = balanced_data["TenYearCHD"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42, shuffle=True
)

# Now Preprocessing
# Standard Scaling
# fitting the train and transforming the test
from sklearn.preprocessing import StandardScaler, QuantileTransformer

scaler = StandardScaler()

# For comparison i did use quantile too ,
qt = QuantileTransformer(
    output_distribution="normal"
)  # Output Distribution removes the outliers

# Standard Scaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print(f'Standard_Scaler_Train{pd.DataFrame(X_train_scaled)}')
# print(f'Standars_Scaler_Test{pd.DataFrame(X_test_scaled)}')

# Quantile Xmer here is not that good
# As its true it will remove the outliers but also creates the problem of over fit
# For sake of program i will go with Standard Scaler
"""X_train_qt=qt.fit_transform(X_train)
X_test__qt=qt.transform(X_test)

print(f'Quantile Scaler Train: {pd.DataFrame(X_train_qt)}')
print(f'Quantile Scale Test: {pd.DataFrame(X_test__qt)}')"""

# Matpltlib
# This is not for the main program but only for thecomparison os standard and Quantile Xmer here
import matplotlib as plt

"""def plot_feature_dist(X, title):
    plt.figure(figsize=(15, 4))
    for i in range(3):  # first 3 features
        plt.subplot(1, 3, i+1)
        plt.hist(X[:, i], bins=30)
        plt.title(f"{title} - Feature {i}")
    plt.tight_layout()
    plt.show()

plot_feature_dist(X_train, "Original")
plot_feature_dist(X_train_scaled, "Standard Scaled")
plot_feature_dist(X_train_qt, "Quantile Transformed")"""

# Metrics : Giving Weiths to the features or colums??
# Importing the Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# Defining the list of classifiers so that thhey can be passed at same time
# The purpose is to stream line and comapre different acuracy by individual models
# Then selecting the optimal
classifier_models = [
    LogisticRegression(),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(
        algorithm="SAMME"
    ),  # the alrithm in adaboost by default is SAMME but defining it doest show the error
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    SVC(),
    XGBClassifier(),
    GaussianNB(),
]
# Creating empty list so then can be append the individual data here
# Train and Evaluate each Calassifier
# List to collect each row of results
results_list = []

# Train and evaluate each classifier
for clf in classifier_models:
    clf_name = clf.__class__.__name__
    clf.fit(X_train_scaled, y_train)
    y_prediction = clf.predict(X_test_scaled)
    # Print the name in Green for better view
    # FOr Terminal result printing
    # print("v" * 50)
    # print(f'\033[92m{clf_name}\033[0m')  # <- this resets the color right after the name
    # print("^" * 50)
    # print()

    # To calculate the accuracy:
    accuracy = accuracy_score(y_test, y_prediction)
    # print(f'{clf_name} accuracy: {accuracy}')
    # print("==" * 50)

    # For Precision
    precision = precision_score(y_test, y_prediction, average="binary")
    # print(f'{clf_name} precision: {precision}')
    # print("=" * 50)
    # Classification report
    # print(f'Classification Report For: {clf_name}')
    # print(f'{classification_report(y_test, y_prediction)}')
    # print("=" * 50)

    # Confusion Matrix:
    # print(f'Confusion Matrix For: {clf_name}')
    # print(f'{confusion_matrix(y_test, y_prediction)}')
    # print("=" * 50)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_prediction)
    report = classification_report(y_test, y_prediction, output_dict=True)
    f1 = report["weighted avg"]["f1-score"]
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]

    # Append result as a dictionary
    results_list.append(
        {
            "Model": clf_name,
            "Accuracy": accuracy,
            "F1-Score": f1,
            "Precision": precision,
            "Recall": recall,
        }
    )

# Create the final DataFrame from the list of results
result = (
    pd.DataFrame(results_list)
    .sort_values(by="Accuracy", ascending=False)
    .reset_index(drop=True)
)
# Show results
print(f"\033[92m{result}\033[0m")
best_model = result.reset_index(drop=True).iloc[0]["Model"]
# Get the name of the best model
print("#" * 69)
# best_model_name = result.reset_index(drop=True).iloc[0]['Model']
print(f"\033[92mThe name the the highest accuracy model is :{best_model}\033[0m")
print("#" * 69)


# Match the name with the actual trained model instance
best_model_instance = None
for clf in classifier_models:
    if clf.__class__.__name__ == best_model:
        best_model_instance = clf
        break

# print("+"*80)
# print(f'\033[92m     Best Model is : {best_model_name}\033[0m')
# print("+"*80)
# print(f'\033[92m     Best Model is :\n{best_model}\033[0m')  # <- this resets the color right after the name

# Saving the file
# Pickle is python inbuild library
# Pickle = file io just for binary

import pickle

pickle.dump(best_model_instance, open(r"model/best_model.pkl", "wb"))
pickle.dump(scaler, open(r"model/scaler.pkl", "wb"))

# Loading the pickle saved file
# For best model
"""with open(r'model/best_model.pkl','rb')as file:
    best_model_instance=pickle.load(file)
#For scaler
with open(r'model/scaler.pkl','rb')as file:
    scaler=pickle.load(file)"""
# print(dataset.head(3))


def final_predict(
    best_model,
    scaler,
    male,
    age,
    currentSmoker,
    cigsPerDay,
    BPMeds,
    prevalentStroke,
    prevalentHyp,
    diabetes,
    totChol,
    sysBP,
    diaBP,
    BMI,
    heartRate,
    glucose,
):
    male_encoded = 1 if male.casefold() == "male" else 0

    currentSmoker_encoded = 1 if currentSmoker.casefold() == "yes" else 0

    BPMeds_encoded = 1 if BPMeds.casefold() == "yes" else 0

    prevalentStroke_encoded = 1 if prevalentStroke.casefold() == "yes" else 0

    prevalentHyp_encoded = 1 if prevalentHyp.casefold() == "yes" else 0

    diabetes_encoded = 1 if diabetes.casefold() == "yes" else 0

    feature = np.array(
        [
            [
                male_encoded,
                age,
                currentSmoker_encoded,
                cigsPerDay,
                BPMeds_encoded,
                prevalentStroke_encoded,
                prevalentHyp_encoded,
                diabetes_encoded,
                totChol,
                sysBP,
                diaBP,
                BMI,
                heartRate,
                glucose,
            ]
        ]
    )

    scaled_feature = scaler.transform(feature)

    predictive_result = best_model.predict(scaled_feature)

    return predictive_result[0]


# test 1:
male = "female"
age = 56.00
currentSmoker = "yes"
cigsPerDay = 3.00
BPMeds = "no"
prevalentStroke = "no"
prevalentHyp = "yes"
diabetes = "no"
totChol = 285.00
sysBP = 145.00
diaBP = 100.00
BMI = 30.14
heartRate = 80.00
glucose = 86.00


test_1 = final_predict(
    best_model_instance,
    scaler,
    male,
    age,
    currentSmoker,
    cigsPerDay,
    BPMeds,
    prevalentStroke,
    prevalentHyp,
    diabetes,
    totChol,
    sysBP,
    diaBP,
    BMI,
    heartRate,
    glucose,
)

print("-" * 69)
print("Test  1")
if test_1 == 1:
    print("\033[1;91m Bruh, You are cooked 💀\033[0m")
    if age <= 40:
        print("They sent Batman to visit you.\n He just came to say goodbye.")
    else:
        print(
            "💀 Yep... definitely not your lucky day.\n Maybe start writing that bucket list?"
        )
else:
    print(
        "\033[92m😌 All good. But I’m just code written by someone who binged Python tutorials at 3AM, so... proceed with caution.\033[0m"
    )


# Test2
male = "female"
age = 63.0
currentSmoker = "yes"
cigsPerDay = 3.0
BPMeds = "no"
prevalentStroke = "no"
prevalentHyp = "yes"
diabetes = "no"
totChol = 267.0
sysBP = 156.5
diaBP = 92.5
BMI = 27.1
heartRate = 60.0
glucose = 79.0
print("=" * 69)
test_2 = final_predict(
    best_model_instance,
    scaler,
    male,
    age,
    currentSmoker,
    cigsPerDay,
    BPMeds,
    prevalentStroke,
    prevalentHyp,
    diabetes,
    totChol,
    sysBP,
    diaBP,
    BMI,
    heartRate,
    glucose,
)
print("-" * 69)
print("Test  2")
if test_2 == 1:
    print("\033[1;91m Bruh, You are cooked 💀\033[0m")
    if age <= 18:
        print(
            "\033[1;91mThey sent Batman to visit you.\n He just came to say goodbye.\033[0m"
        )
    else:
        print(
            "\033[1;91m💀 Yep... definitely not your lucky day.\n Maybe start writing that bucket list?\033[0m"
        )
else:
    print(
        "\033[92m😌 All good. But I’m just code written by someone who binged Python tutorials at 3AM, so... proceed with caution.\033[0m"
    )
