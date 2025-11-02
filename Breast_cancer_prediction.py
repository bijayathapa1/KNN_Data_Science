import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict_breast_cancer

st.title("Breast Cancer Prediction App")



radius_mean = st.number_input("Radius Mean")
texture_mean = st.number_input("Texture Mean")
perimeter_mean = st.number_input("Perimeter Mean")
area_mean = st.number_input("Area Mean")
smoothness_mean = st.number_input("Smoothness Mean")
compactness_mean = st.number_input("Compactness Mean")
concavity_mean = st.number_input("Concavity Mean")
concave_points_mean = st.number_input("Concave Points Mean")
symmetry_mean = st.number_input("Symmetry Mean")
fractal_dimension_mean = st.number_input("Fractal Dimension Mean")

radius_se = st.number_input("Radius SE")
texture_se = st.number_input("Texture SE")
perimeter_se = st.number_input("Perimeter SE")
area_se = st.number_input("Area SE")
smoothness_se = st.number_input("Smoothness SE")
compactness_se = st.number_input("Compactness SE")
concavity_se = st.number_input("Concavity SE")
concave_points_se = st.number_input("Concave Points SE")
symmetry_se = st.number_input("Symmetry SE")
fractal_dimension_se = st.number_input("Fractal Dimension SE")

radius_worst = st.number_input("Radius Worst")
texture_worst = st.number_input("Texture Worst")
perimeter_worst = st.number_input("Perimeter Worst")
area_worst = st.number_input("Area Worst")
smoothness_worst = st.number_input("Smoothness Worst")
compactness_worst = st.number_input("Compactness Worst")
concavity_worst = st.number_input("Concavity Worst")
concave_points_worst = st.number_input("Concave Points Worst")
symmetry_worst = st.number_input("Symmetry Worst")
fractal_dimension_worst = st.number_input("Fractal Dimension Worst")

btn=st.button("Predict Breast Cancer")
inference_data = np.array([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                 concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se,
                 perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
                 radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
                 concave_points_worst, symmetry_worst, fractal_dimension_worst])

if btn:
    result=predict_breast_cancer(inference_data)
    if result[0]=='M':
        st.error("The patient is likely to have Breast Cancer.")
    else:
        st.success("The patient is unlikely to have Breast Cancer.")

