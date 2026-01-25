import streamlit as st
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import itertools

# --- 1. Configuration & Dictionaries ---
st.set_page_config(page_title="Census PUMS Clustering", layout="wide")

# Map of State Names to FIPS Codes (Simplified list)
STATE_FIPS = {
    "Alabama": "01", "California": "06", "Colorado": "08", 
    "Florida": "12", "Georgia": "13", "Illinois": "17", 
    "New York": "36", "Texas": "48", "Virginia": "51", "Washington": "53"
}

# Mapping PUMS Codes to Human Readable Labels
# Source: Census PUMS Data Dictionary
EDUCATION_MAP = {
    '16': 'High School Diploma', '17': 'GED', '18': 'Some College (<1yr)',
    '19': 'Some College (1+yr)', '20': 'Associate Degree', 
    '21': 'Bachelor\'s Degree', '22': 'Master\'s Degree', 
    '23': 'Professional Degree', '24': 'Doctorate'
}

# 1: Married, 2: Other Family, 3: Non-Family, etc. (Simplified)
HHT_MAP = {
    '1': 'Married Couple', '2': 'Male Householder (No Spouse)',
    '3': 'Female Householder (No Spouse)', '4': 'Non-Family (Male)',
    '5': 'Non-Family (Female)', '6': 'Group Quarters', '7': 'Group Quarters'
}

# --- 2. Helper Functions ---

@st.cache_data
def fetch_pums_data(state_fips, api_key):
    """
    Fetches PUMS data for Age, Income, Household Type, and Education.
    Uses the 2022 ACS 1-Year PUMS API.
    """
    # Variables:
    # AGEP: Age
    # PINCP: Personal Income
    # HHT: Household/Family Type
    # SCHL: Educational Attainment
    base_url = "https://api.census.gov/data/2022/acs/acs1/pums"
    
    params = {
        "get": "AGEP,PINCP,HHT,SCHL",
        "for": f"state:{state_fips}",
        "key": api_key
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # The first row is headers
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Convert columns to numeric
        cols = ['AGEP', 'PINCP', 'HHT', 'SCHL']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        return df.dropna()
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def process_data(df):
    """
    Transforms data: Bins age, decodes labels for display.
    """
    # 1. Filter out weird values (e.g. Income < 0 or Null education)
    df = df[(df['PINCP'] >= 0) & (df['AGEP'] > 18)]
    
    # 2. Transform Age into Ranges
    bins = [18, 30, 45, 60, 100]
    labels = ['18-30', '31-45', '46-60', '60+']
    df['Age Range'] = pd.cut(df['AGEP'], bins=bins, labels=labels)
    
    # 3. Add Readable Columns for Display
    df['Education Level'] = df['SCHL'].astype(str).map(EDUCATION_MAP).fillna('Other/Less than HS')
    df['Household Type'] = df['HHT'].astype(str).map(HHT_MAP).fillna('Unknown')
    
    return df

# --- 3. Main Streamlit App ---

st.title("🇺🇸 Census PUMS Clustering Analysis")
st.markdown("""
This app fetches **Public Use Microdata Sample (PUMS)** data directly from the Census API. 
It clusters individuals based on **Age, Income, Household Type, and Education**.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

# Check for API Key
if "CENSUS_API_KEY" in st.secrets:
    api_key = st.secrets["CENSUS_API_KEY"]
else:
    st.sidebar.error("CENSUS_API_KEY not found in secrets.")
    st.stop()

selected_state = st.sidebar.selectbox("Select US State", list(STATE_FIPS.keys()))
n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 8, 3)

# --- Data Loading ---
if st.sidebar.button("Fetch & Analyze Data"):
    with st.spinner(f"Fetching PUMS data for {selected_state}... (This may take a moment)"):
        fips = STATE_FIPS[selected_state]
        raw_df = fetch_pums_data(fips, api_key)
        
    if not raw_df.empty:
        df = process_data(raw_df)
        
        # --- Clustering ---
        # Select features for clustering (Using the numeric codes)
        features = ['AGEP', 'PINCP', 'HHT', 'SCHL']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(scaled_features)
        
        st.success(f"Successfully loaded {len(df)} records and created {n_clusters} clusters.")
        
        # --- Visualization Section ---
        st.divider()
        st.subheader("📊 Cluster Visualizations")
        
        # Define features for plotting (User Friendly Names)
        plot_vars = {
            'Age': 'AGEP',
            'Income': 'PINCP',
            'Education Code': 'SCHL',
            'Household Code': 'HHT'
        }
        
        # Create list of all combinations
        combinations = list(itertools.combinations(plot_vars.keys(), 2))
        combo_labels = [f"{x} vs {y}" for x, y in combinations]
        
        # User Selection
        selected_combo = st.selectbox("Select Graph to Show:", combo_labels)
        
        # Find the keys for the selected combo
        x_label, y_label = next(
            (x, y) for x, y in combinations if f"{x} vs {y}" == selected_combo
        )
        
        # Plot
        fig = px.scatter(
            df.sample(min(2000, len(df))), # Sample for performance if needed
            x=plot_vars[x_label], 
            y=plot_vars[y_label],
            color='Cluster',
            title=f"Clusters: {x_label} vs {y_label}",
            hover_data=['Age Range', 'Education Level', 'Household Type'],
            color_continuous_scale=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Summary Table ---
        st.divider()
        st.subheader("📋 Cluster Summary")
        st.markdown("Average values and most common categories per cluster.")
        
        # Group by Cluster
        summary = df.groupby('Cluster').agg({
            'PINCP': 'mean',
            'AGEP': 'mean',
            'Education Level': lambda x: x.mode()[0] if not x.mode().empty else "N/A",
            'Household Type': lambda x: x.mode()[0] if not x.mode().empty else "N/A",
            'Age Range': lambda x: x.mode()[0] if not x.mode().empty else "N/A"
        }).reset_index()
        
        # Rename columns for readability
        summary = summary.rename(columns={
            'PINCP': 'Avg Income ($)',
            'AGEP': 'Avg Age',
            'Education Level': 'Most Common Education',
            'Household Type': 'Most Common Household',
            'Age Range': 'Most Common Age Group'
        })
        
        # Format Income
        summary['Avg Income ($)'] = summary['Avg Income ($)'].apply(lambda x: f"${x:,.2f}")
        summary['Avg Age'] = summary['Avg Age'].round(1)
        
        st.table(summary)
        
    else:
        st.warning("No data found or API limit reached. Try a different state or check API key.")
else:
    st.info("Select a state and click 'Fetch & Analyze Data' to begin.")
