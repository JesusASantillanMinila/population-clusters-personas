import streamlit as st
import pandas as pd
from census import Census
from us import states
import google.generativeai as genai
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# --- Configuration & Setup ---
st.set_page_config(page_title="Census Cluster Persona Generator", layout="wide")

# check for secrets
if "CENSUS_API_KEY" not in st.secrets or "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing API Keys. Please ensure CENSUS_API_KEY and GOOGLE_API_KEY are in .streamlit/secrets.toml")
    st.stop()

CENSUS_API_KEY = st.secrets["CENSUS_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# --- Helper Functions ---

@st.cache_data
def get_census_data(api_key, state_fips, year=2021):
    """
    Fetches county-level data for:
    - Median Household Income (B19013_001E)
    - Median Age (B01002_001E)
    - Total Population (B01003_001E)
    """
    c = Census(api_key, year=year)
    
    # Fetching data for all counties in the specified state
    data = c.acs5.state_county(
        fields=('NAME', 'B19013_001E', 'B01002_001E', 'B01003_001E'),
        state_fips=state_fips,
        county_fips=Census.ALL
    )
    
    df = pd.DataFrame(data)
    
    # Rename columns for clarity
    df = df.rename(columns={
        'B19013_001E': 'Median_Income',
        'B01002_001E': 'Median_Age',
        'B01003_001E': 'Population',
        'NAME': 'County_Name'
    })
    
    # Cleaning: Drop rows with missing data (often -666666666 in Census data)
    df = df[df['Median_Income'] > 0]
    df = df[df['Median_Age'] > 0]
    df = df.dropna()
    
    return df

def generate_personas(cluster_stats):
    """
    Uses Gemini to generate creative personas based on cluster statistics.
    """
    model = genai.GenerativeModel('gemini-pro')
    
    # Construct a prompt based on the cluster averages
    prompt = f"""
    You are a demographic expert. I have clustered counties in a US state based on Median Income, Age, and Population.
    Here are the statistics for each cluster:
    
    {cluster_stats.to_string()}
    
    For EACH cluster, strictly provide a response in the following format:
    
    Cluster [ID]:
    Name: [Creative Persona Name, e.g., "Retiring Wealthy", "Young Up-and-Comers"]
    Description: [A 2-sentence description of who lives here based on the data]
    """
    
    with st.spinner("Asking Gemini to analyze demographics..."):
        response = model.generate_content(prompt)
    
    return response.text

# --- Sidebar UI ---
st.sidebar.header("Configuration")

# State Selection
all_states = {state.name: state.fips for state in states.STATES}
selected_state_name = st.sidebar.selectbox("Select State", list(all_states.keys()))
selected_state_fips = all_states[selected_state_name]

# Clustering Parameters
num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=6, value=3)
min_pop = st.sidebar.number_input("Minimum County Population", min_value=0, value=10000, step=1000)

# --- Main App Logic ---

st.title(f"📍 Demographic Clusters for {selected_state_name}")
st.markdown("This tool groups counties by Age, Income, and Population, then uses AI to describe the resulting personas.")

# 1. Load Data
try:
    df = get_census_data(CENSUS_API_KEY, selected_state_fips)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# 2. Filter Data
filtered_df = df[df['Population'] >= min_pop].copy()
st.write(f"Analyzing {len(filtered_df)} counties (filtered out {len(df) - len(filtered_df)} below population threshold).")

if len(filtered_df) < num_clusters:
    st.error("Not enough counties remain after filtering to perform clustering. Try lowering the population threshold.")
    st.stop()

# 3. Perform Clustering
# Select features for clustering
features = filtered_df[['Median_Income', 'Median_Age']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
filtered_df['Cluster'] = kmeans.fit_predict(features_scaled)

# 4. Calculate Cluster Statistics for AI
cluster_stats = filtered_df.groupby('Cluster')[['Median_Income', 'Median_Age', 'Population']].mean().reset_index()
cluster_stats['Median_Income'] = cluster_stats['Median_Income'].round(0)
cluster_stats['Median_Age'] = cluster_stats['Median_Age'].round(1)
cluster_stats['Population'] = cluster_stats['Population'].round(0)

# 5. Generate AI Personas
if st.button("Generate AI Personas"):
    ai_response = generate_personas(cluster_stats)
    
    # Layout: Graph on top, Personas below
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Cluster Visualization")
        fig = px.scatter(
            filtered_df,
            x='Median_Age',
            y='Median_Income',
            size='Population',
            color='Cluster',
            hover_name='County_Name',
            title=f"Income vs Age by Cluster (Size = Population)",
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Data Overview")
        st.dataframe(cluster_stats.style.format({
            "Median_Income": "${:,.0f}",
            "Median_Age": "{:.1f}",
            "Population": "{:,.0f}"
        }), use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 Gemini Generated Personas")
    st.markdown(ai_response)

else:
    st.info("Click the button above to generate clusters and AI analysis.")
