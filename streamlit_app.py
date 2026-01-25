import streamlit as st
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import itertools 

# --- 1. Configuration & Dictionaries ---
st.set_page_config(page_title="Census PUMS Clustering", layout="wide")

# Map of State Names to FIPS Codes
STATE_FIPS = {
    "Alabama": "01", "California": "06", "Colorado": "08", 
    "Florida": "12", "Georgia": "13", "Illinois": "17", 
    "New York": "36", "Texas": "48", "Virginia": "51", "Washington": "53"
}

# Mapping PUMS Codes to Human Readable Labels
EDUCATION_MAP = {
    '16': 'High School Diploma', '17': 'GED', '18': 'Some College (<1yr)',
    '19': 'Some College (1+yr)', '20': 'Associate Degree', 
    '21': 'Bachelor\'s Degree', '22': 'Master\'s Degree', 
    '23': 'Professional Degree', '24': 'Doctorate'
}

# Define a logical sort order for Education for the graph axis
EDU_ORDER = list(EDUCATION_MAP.values()) + ['Other/Less than HS']

# 1: Married, 2: Other Family, 3: Non-Family, etc.
HHT_MAP = {
    '1': 'Married Couple', '2': 'Male Householder (No Spouse)',
    '3': 'Female Householder (No Spouse)', '4': 'Non-Family (Male)',
    '5': 'Non-Family (Female)', '6': 'Group Quarters', '7': 'Group Quarters'
}

# --- 2. Helper Functions ---

@st.cache_data
def fetch_pums_data(state_fips, api_key):
    """Fetches PUMS data for Age, Income, Household Type, and Education."""
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
        df = pd.DataFrame(data[1:], columns=data[0])
        cols = ['AGEP', 'PINCP', 'HHT', 'SCHL']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def process_data(df):
    """Transforms data: Bins age, decodes labels."""
    df = df[(df['PINCP'] >= 0) & (df['AGEP'] > 18)]
    bins = [18, 30, 45, 60, 100]
    labels = ['18-30', '31-45', '46-60', '60+']
    df['Age Range'] = pd.cut(df['AGEP'], bins=bins, labels=labels)
    # Create readable columns
    df['Education Level'] = df['SCHL'].astype(str).map(EDUCATION_MAP).fillna('Other/Less than HS')
    df['Household Type'] = df['HHT'].astype(str).map(HHT_MAP).fillna('Unknown')
    return df

# --- 3. Main Streamlit App ---

st.title("🇺🇸 Census PUMS Clustering Analysis")
st.markdown("Fetch PUMS data, create clusters, and visualize population segments.")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

if "CENSUS_API_KEY" in st.secrets:
    api_key = st.secrets["CENSUS_API_KEY"]
else:
    st.sidebar.error("CENSUS_API_KEY not found in secrets.")
    st.stop()

selected_state = st.sidebar.selectbox("Select US State", list(STATE_FIPS.keys()))
n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 8, 3)

# --- Initialize Session State ---
if 'data' not in st.session_state:
    st.session_state['data'] = None

# --- Data Loading Logic ---
if st.sidebar.button("Fetch & Analyze Data"):
    with st.spinner(f"Fetching PUMS data for {selected_state}..."):
        fips = STATE_FIPS[selected_state]
        raw_df = fetch_pums_data(fips, api_key)
        
        if not raw_df.empty:
            df = process_data(raw_df)
            
            # Clustering (Use numeric codes for the math, but we plot with labels later)
            features = ['AGEP', 'PINCP', 'HHT', 'SCHL']
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[features])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(scaled_features)
            
            # SAVE TO SESSION STATE
            st.session_state['data'] = df
            st.success(f"Loaded {len(df)} records. Clusters created!")
        else:
            st.warning("No data found.")

# --- Display Logic (Runs if data exists in state) ---
if st.session_state['data'] is not None:
    df = st.session_state['data']
    
    st.divider()
    st.subheader("📊 Cluster Visualizations")
    
    # Updated: Map UI choices to HUMAN READABLE columns, not codes
    plot_vars = {
        'Age': 'AGEP',
        'Income': 'PINCP',
        'Education': 'Education Level',   # Changed from 'SCHL'
        'Household Type': 'Household Type' # Changed from 'HHT'
    }
    
    combinations = list(itertools.combinations(plot_vars.keys(), 2))
    combo_labels = [f"{x} vs {y}" for x, y in combinations]
    
    selected_combo = st.selectbox("Select Graph to Show:", combo_labels)
    
    x_label, y_label = next(
        (x, y) for x, y in combinations if f"{x} vs {y}" == selected_combo
    )
    
    # Prepare data for plotting
    plot_df = df.sample(min(2000, len(df))).copy()
    
    # IMPORTANT: Convert Cluster to string to ensure DISCRETE Color Legend (not gradient)
    plot_df['Cluster'] = plot_df['Cluster'].astype(str)
    
    fig = px.scatter(
        plot_df, 
        x=plot_vars[x_label], 
        y=plot_vars[y_label],
        color='Cluster',
        title=f"Clusters: {x_label} vs {y_label}",
        hover_data=['Age Range', 'Education Level', 'Household Type'],
        # Define readable axis titles
        labels={
            'AGEP': 'Age (Years)',
            'PINCP': 'Annual Income ($)',
            'Education Level': 'Education Degree',
            'Household Type': 'Household Composition',
            'Cluster': 'Cluster Group'
        },
        # Ensure Education sorts logically (HS -> PhD), not alphabetically
        category_orders={'Education Level': EDU_ORDER}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Summary Table ---
    st.divider()
    st.subheader("📋 Cluster Summary")
    
    summary = df.groupby('Cluster').agg({
        'PINCP': 'mean',
        'AGEP': 'mean',
        'Education Level': lambda x: x.mode()[0] if not x.mode().empty else "N/A",
        'Household Type': lambda x: x.mode()[0] if not x.mode().empty else "N/A",
        'Age Range': lambda x: x.mode()[0] if not x.mode().empty else "N/A"
    }).reset_index()
    
    summary = summary.rename(columns={
        'PINCP': 'Avg Income ($)',
        'AGEP': 'Avg Age',
        'Education Level': 'Most Common Education',
        'Household Type': 'Most Common Household',
        'Age Range': 'Most Common Age Group'
    })
    
    summary['Avg Income ($)'] = summary['Avg Income ($)'].apply(lambda x: f"${x:,.2f}")
    summary['Avg Age'] = summary['Avg Age'].round(1)
    
    st.table(summary)

elif st.session_state['data'] is None:
    st.info("Select a state and click 'Fetch & Analyze Data' to begin.")
