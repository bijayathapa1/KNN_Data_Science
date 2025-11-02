import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNeighborClassifier
import joblib



breast_cancer_data = pd.read_csv(r'C:\Users\user\Downloads\breast-cancer.csv')
breast_cancer_data.drop(columns=['id'], inplace=True)

label_encoder = LabelEncoder()
breast_cancer_data['diagnosis'] = label_encoder.fit_transform(breast_cancer_data['diagnosis'])

X_bc = breast_cancer_data.drop(columns=['diagnosis'])
y_bc = breast_cancer_data['diagnosis']

X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)

scaler_bc = StandardScaler()
X_train_bc = scaler_bc.fit_transform(X_train_bc)
X_test_bc =  scaler_bc.transform(X_test_bc)

knn_bc_model = KNeighborClassifier(n_neighbors=5, metric='euclidean')
knn_bc_model.fit(X_train_bc, y_train_bc)

joblib.dump(knn_bc_model, 'logistic_regression_model.joblib')
joblib.dump(scaler_bc, 'Breast_cancer_scaler.joblib')
joblib.dump(label_encoder, 'Breast_cancer_label_encoder.joblib')
