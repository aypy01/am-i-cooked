# Am I Cooked? 
## Heart Disease Prediction Web App

This project combines a machine learning model for predicting 10-year Coronary Heart Disease (CHD) risk with a user-friendly Flask web application — and a little bit of dark humor.

Ever looked at your lifestyle and wondered *“Am I cooked?”* Well, now there’s a web app that can tell you — based on science, not just your last late-night pizza binge.

---

## 🩺 Introduction

This application uses the **Framingham Heart Study dataset** to predict the risk of CHD. It’s a blend of data preprocessing, model training, and front-end flavor — all served with some cheeky personality. You input your health data, and it tells you if your heart’s chilling or on thin ice (figuratively, of course).

---

## 📊 Dataset

The dataset, `framingham.csv`, is sourced from [Kaggle](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset). It includes the following features:

- `male`: Male (1) or Female (0)  
- `age`: Age of the patient  
- `currentSmoker`: Currently smoking (1 = yes, 0 = no)  
- `cigsPerDay`: Avg. cigarettes smoked per day  
- `BPMeds`: On blood pressure medication  
- `prevalentStroke`: History of stroke  
- `prevalentHyp`: Hypertension  
- `diabetes`: Diabetes status  
- `totChol`: Total cholesterol  
- `sysBP`: Systolic BP  
- `diaBP`: Diastolic BP  
- `BMI`: Body Mass Index  
- `heartRate`: Heart rate  
- `glucose`: Glucose level  
- `education`: Education level  
- `TenYearCHD`: Target variable (1 = CHD in 10 years, 0 = no)

---

## ⚙️ Dependencies

- `Flask`: Web framework  
- `numpy`: Numerical computation  
- `pandas`: Data manipulation  
- `scikit-learn`: Machine learning  
- `xgboost`: Gradient boosting  
- `matplotlib`: For visualizations (if needed)  
- `pickle`: For model serialization  

All dependencies can be installed using the `requirements.txt`.

---

## 💻 How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/am-i-cooked.git
cd am-i-cooked
