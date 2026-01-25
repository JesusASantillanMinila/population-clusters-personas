import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from census import Census
from us import states
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# --- 1. CONFIGURATION & MAPPINGS ---
st.set_page_config(page_title="Census PUMS Clustering", layout="wide")

# Map FIPS codes to State Names
STATE_FIPS = {state.fips: state.name for state in states.STATES}
STATE_NAME_TO_FIPS = {v: k for k, v in STATE_FIPS.items()}

# Education Level Mapping (SCHL Code -> Readable Name)
# Based on ACS Data Dictionary
EDUCATION_MAP = {
    'bb': 'N/A (under 3)',
    '01': 'No schooling',
    '02': 'Nursery to 4th grade',
    '03': '5th or 6th grade',
    '04': '7th or 8th grade',
    '05': '9th grade',
    '06': '10th grade',
    '07': '11th grade',
    '08': '12th grade (no diploma)',
    '09': 'High School Diploma',
    '10': 'GED or alternative',
    '11': 'Some college, <1 year',
    '12': 'Some college, >1 year',
    '13': 'Associate\'s degree',
    '14': 'Associate\'s degree',
    '15': 'Bachelor\'s degree',
    '16': 'Master\'s degree',
    '17': 'Professional degree',
    '18': 'Doctorate degree',
    '19': 'Grade 12', # Occasionally appears in some cuts
    '20': 'Bachelor\'s degree', # Redundant handling
    '21': 'Master\'s degree',
    '22': 'Professional degree',
    '23': 'Doctorate degree',
    '24': 'Doctorate degree'
}

# Household Type Mapping (HHT Code -> Readable Name)
HHT_MAP = {
    '1': 'Married couple household',
    '2': 'Male householder, no spouse',
    '3': 'Female householder, no spouse',
    '4': 'Male living alone',
    '5': 'Male not living alone',
    '6': 'Female living alone',
    '7': 'Female not living alone',
    'b': 'N/A (Group Quarters)'
}

# --- 2. DATA FETCHING FUNCTION ---
@st.cache_data
def get_census_data(api_key, state_fips, limit=5000):
    """
    Fetches PUMS data for Age (AGEP), Income (HINCP), 
    Household Type (HHT), and Education (SCHL).
    """
    try:
        c = Census(api_key)
        
        # Fetching variables:
        # AGEP: Age
        # HINCP: Household Income
        # HHT: Household/Family Type
        # SCHL: Educational Attainment
        
        # We target the most recent available ACS 1-Year data (usually 2021 or 2022 depending on lib version)
        # Using 2021 as a safe default for 'census' library stability
        data = c.acs1.state(
            fields=('AGEP', 'HINCP', 'HHT', 'SCHL'),
            state_fips=state_fips,
            year=2021 
        )
        
        df = pd.DataFrame(data)
        
        # Convert numeric columns
        df['AGEP'] = pd.to_numeric(df['AGEP'], errors='coerce')
        df['HINCP'] = pd.to_numeric(df['HINCP'], errors='coerce')
        
        # Drop rows with missing core data (often due to vacant houses or group quarters)
        df = df.dropna(subset=['AGEP', 'HINCP', 'HHT', 'SCHL'])
        
        # Filter out negative income if preferred, or keep for debt analysis
        # Filtering empty strings or bad codes
        df = df[df['HHT'] != 'b']
        
        # Limit sample size for performance in a demo app
        if len(df) > limit:
            df = df.sample(limit, random_state=42)
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- 3. MAIN APPLICATION ---

st.title("📊 Census PUMS Clustering Explorer")
st.markdown("Cluster population data based on **Age**, **Income**, **Household Type**, and **Education**.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Census API Key", type="password", help="Get one at api.census.gov")
selected_state = st.sidebar.selectbox("Select State", list(STATE_NAME_TO_FIPS.keys()))
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=8, value=3)

# --- VISUALIZATION CONTROLS ---
st.sidebar.subheader("Graph Settings")
x_axis = st.sidebar.selectbox("X Axis", ['Age', 'Income', 'Education Code', 'Household Type Code'])
y_axis = st.sidebar.selectbox("Y Axis", ['Income', 'Age', 'Education Code', 'Household Type Code'])

if not api_key:
    st.warning("Please enter your Census API Key in the sidebar to proceed.")
    st.stop()

# --- FETCH AND PROCESS DATA ---
with st.spinner(f"Fetching data for {selected_state}..."):
    raw_df = get_census_data(api_key, STATE_NAME_TO_FIPS[selected_state])

if raw_df.empty:
    st.error("No data found. Please check your API key or try another state.")
else:
    # --- DATA PREPROCESSING ---
    # Create readable labels columns for display
    raw_df['Education Level'] = raw_df['SCHL'].map(EDUCATION_MAP).fillna('Unknown')
    raw_df['Household Type'] = raw_df['HHT'].map(HHT_MAP).fillna('Unknown')
    
    # Encode Categoricals for Clustering (Education is Ordinal, HHT is Nominal)
    # 1. Age & Income are already numeric.
    # 2. Education: Map codes to integers (they are roughly ordinal in the source)
    raw_df['SCHL_Code'] = pd.to_numeric(raw_df['SCHL'], errors='coerce').fillna(0)
    
    # 3. HHT: It's nominal. For clustering, we usually One-Hot Encode, 
    # but to keep the dataframe simple for the "X vs Y" scatter plot request,
    # we will use a Label Encoder for the 'cluster input' specifically, or simple numeric mapping.
    # For better clustering, we'll dummy encode HHT.
    
    features = raw_df[['AGEP', 'HINCP', 'SCHL_Code']].copy()
    
    # One-Hot Encode HHT for the algorithm
    hht_dummies = pd.get_dummies(raw_df['HHT'], prefix='HHT')
    cluster_input = pd.concat([features, hht_dummies], axis=1)
    
    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_input)
    
    # --- CLUSTERING ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    raw_df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # --- MAPPING SELECTION FOR PLOTTING ---
    # Map user selection to dataframe columns
    col_map = {
        'Age': 'AGEP',
        'Income': 'HINCP',
        'Education Code': 'SCHL_Code',
        'Household Type Code': 'HHT'
    }
    
    # --- VISUALIZATION ---
    st.subheader(f"{selected_state}: {x_axis} vs {y_axis} by Cluster")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # If using categorical columns on axes, ensure they are sorted/numeric for plotting
    plot_x = col_map[x_axis]
    plot_y = col_map[y_axis]
    
    # Use seaborn for a clean scatter plot
    sns.scatterplot(
        data=raw_df, 
        x=plot_x, 
        y=plot_y, 
        hue='Cluster', 
        palette='viridis', 
        alpha=0.6,
        ax=ax,
        s=50
    )
    
    plt.title(f"Clustering Results: {x_axis} vs {y_axis}")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot(fig)
    
    # --- SUMMARY TABLE ---
    st.subheader("Cluster Summaries")
    
    # Calculate modes for categorical data and means for numerical
    summary = raw_df.groupby('Cluster').agg({
        'AGEP': 'mean',
        'HINCP': 'mean',
        'Education Level': lambda x: x.mode()[0] if not x.mode().empty else 'N/A',
        'Household Type': lambda x: x.mode()[0] if not x.mode().empty else 'N/A',
        'Cluster': 'count' # To get size
    }).rename(columns={'Cluster': 'Count'})
    
    # Format the summary for readability
    summary['AGEP'] = summary['AGEP'].round(1).astype(str) + " years"
    summary['HINCP'] = summary['HINCP'].apply(lambda x: f"${x:,.0f}")
    summary = summary.rename(columns={
        'AGEP': 'Average Age',
        'HINCP': 'Average Income',
        'Education Level': 'Most Common Education',
        'Household Type': 'Most Common Household'
    })
    
    st.table(summary)
    
    # --- RAW DATA EXPANDER ---
    with st.expander("View Raw Data Sample"):
        st.dataframe(raw_df[['AGEP', 'HINCP', 'Education Level', 'Household Type', 'Cluster']].head(100))
