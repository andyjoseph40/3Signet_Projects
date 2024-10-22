import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set up custom CSS for a beautiful background
page_bg_img = '''
<style>
    body {
        background-image: url("https://img.freepik.com/free-vector/abstract-big-data-technology-concept-background-design_1017-22911.jpg");
        background-size: cover;
    }
    .stApp {
        background: rgba(255, 255, 255, 0.9);  /* Optional: Light background on content */
    }
</style>
'''

# Apply the background image using custom HTML
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('dropout.csv')
    return data

df2 = load_data()

# Sidebar for filters
st.sidebar.header('Filter Options')

# Dropdown (selectbox) for Target filter with 'None' option to cancel selection
selected_target = st.sidebar.selectbox('Select Target:', ['None'] + list(df2['Target'].unique()))

# Dropdown for Age Category filter with 'None' option
selected_age_category = st.sidebar.selectbox('Select Age Category:', ['None'] + list(df2['Age_category'].unique()))

# Dropdown for Curricular Units 1st Sem Grade Category filter with 'None' option
selected_units1st_sem_category = st.sidebar.selectbox('Select Curricular Units 1st Sem (Grade) Category:', 
                                                      ['None'] + list(df2['Curricular units 1st sem (grade)_category'].unique()))

# Dropdown for Curricular Units 2nd Sem Grade Category filter with 'None' option
selected_units2nd_sem_category = st.sidebar.selectbox('Select Curricular Units 2nd Sem (Grade) Category:', 
                                                      ['None'] + list(df2['Curricular units 2nd sem (grade)_category'].unique()))

# Apply filters based on the dropdown selections
filtered_data = df2.copy()

if selected_target != 'None':
    filtered_data = filtered_data[filtered_data['Target'] == selected_target]

if selected_age_category != 'None':
    filtered_data = filtered_data[filtered_data['Age_category'] == selected_age_category]

if selected_units1st_sem_category != 'None':
    filtered_data = filtered_data[filtered_data['Curricular units 1st sem (grade)_category'] == selected_units1st_sem_category]

if selected_units2nd_sem_category != 'None':
    filtered_data = filtered_data[filtered_data['Curricular units 2nd sem (grade)_category'] == selected_units2nd_sem_category]

# Dashboard visuals
st.title('Interactive Data Dashboard')

# PCA Analysis and Visualization
# Select numerical columns for PCA
pca_cols = [
    'Previous qualification (grade)', 
    'Admission grade', 
    'Curricular units 1st sem (grade)', 
    'Curricular units 2nd sem (grade)', 
    'Unemployment rate', 
    'Inflation rate', 
    'GDP'
]

# Standardize the data before applying PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(filtered_data[pca_cols])

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
pca_result = pca.fit_transform(scaled_data)

# Create a new DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# Add the Target column to color the data points by class
pca_df['Target'] = filtered_data['Target']

# Plot the PCA results
st.subheader('PCA - 2 Components')
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Target', data=pca_df, palette='viridis')
plt.title('PCA - 2 Components')
st.pyplot(plt)

# Display explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
st.write(f"Explained variance ratio: {explained_variance_ratio}")

# Plot Admission Grade vs Target
st.subheader('Bar Chart: Admission Grade vs Target')
fig_bar = px.bar(filtered_data, x='Admission grade', y='Target', 
                 color='Target', title="Admission Grade vs Target")
st.plotly_chart(fig_bar)

# Scatter Plot: Admission Grade vs Curricular Units 1st Sem (Grade)
st.subheader('Scatter Plot: Admission Grade vs Curricular Units 1st Sem (Grade)')
fig_scatter = px.scatter(filtered_data, x='Admission grade', y='Curricular units 1st sem (grade)', 
                         color='Target', title="Admission Grade vs Curricular Units 1st Sem (Grade)")
st.plotly_chart(fig_scatter)

# Correlation Heatmap
numeric_columns = ['Previous qualification (grade)', 'Admission grade', 
                   'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 
                   'Unemployment rate', 'Inflation rate', 'GDP']
filtered_numeric_data = filtered_data[numeric_columns]

st.subheader('Correlation Heatmap')
fig, ax = plt.subplots()
sns.heatmap(filtered_numeric_data.corr(), annot=True, ax=ax)
st.pyplot(fig)

# Display filtered data
st.subheader('Filtered Data')
st.write(filtered_data)