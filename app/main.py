import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings



def get_clean_data(): 
    data = pd.read_csv("data/data.csv")
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    data.diagnosis = data.diagnosis.map({'M':1, 'B':0})
    return data 

def add_side_bar(): 
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data() 

    # Define the labels
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    # Add the sliders
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return input_dict 

# Import the scaler
def get_scaled_values(input_dict):
    data = get_clean_data() 

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items(): 
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val)/ (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_dict):
    # Scale the values
    input_data = get_scaled_values(input_dict)

    # Create the radar chart
    fig = go.Figure()

    # Add the traces
    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
                input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
                input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
                input_data['fractal_dimension_mean']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Mean'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
                input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
                input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Standard Error'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                input_data['fractal_dimension_worst']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Worst'
        )
    )

    # Update the layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )

    return fig
 
def add_predictions(input_data): 
    model= pickle.load(open("model/model.pkl", "rb"))
    scaler= pickle.load(open("model/scaler.pkl", "rb"))
    input_array = np.array(list(input_data.values())).reshape(1,-1)
    scaled_input_array = scaler.transform(input_array) 
    prediction = model.predict(scaled_input_array)
    prob_value = model.predict_proba(scaled_input_array)
    benign_prob   = prob_value[0][0] *100 
    malignant_prob = prob_value[0][1]*100  
    
    st.subheader("Interpretations")
    st.write(" the cell cluster is:")
    if prediction[0] == 0: 
        st.write("<span style='color: green; font-weight: bold;'>Benign</span>", unsafe_allow_html=True)
        st.write(f"The chances of being benign is about {benign_prob:.0f}%")
    else: 
        st.write("<span style='color: red; font-weight: bold;'>Malignant</span>", unsafe_allow_html=True)
        st.write(f"The chances of being benign is about {malignant_prob:.0f}%")

    st.write("<span id='notice' style='color: orange; font-weight: bold; font-style: italic; font-size: larger;'>Important Notice!!</span>", unsafe_allow_html=True)
    st.write("While this app can support medical professionals in making diagnoses, it is essential to remember that it is not a replacement for a professional diagnosis.")

def main(): 
    st.set_page_config ( 
        page_title = "Breast Cancer Predictor", 
        page_icon=":female-doctor",
        layout ="wide", 
        initial_sidebar_state="expanded"
    ) 
    input_data = add_side_bar()
    with st.container(): 
        st.title("Breast Cancer  prediction")
        st.write('Welcome to the breast cancer diagnosis app! This tool is your partner in swiftly and accurately diagnosing breast cancer from tissue samples. Powered by advanced machine learning algorithms, it provides rapid assessments on whether a breast mass is benign or malignant based on the data you input. You have two modes of interaction: Automatic Prediction, where you can upload tissue sample measurements for immediate results, and Manual Adjustment, allowing you to fine-tune predictions using intuitive sliders in the sidebar. With this app, you can expedite diagnoses, enhance patient care, and make a real impact in the fight against breast cancer.')    
    
    col1, col2 = st.columns([4,1]) 
    with col1: 
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2: 
        st.write(add_predictions(input_data))
    

if __name__ == '__main__': 
    main() 

