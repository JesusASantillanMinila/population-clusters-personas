import streamlit as st
import pandas as pd
from census import Census
from us import states
from sklearn.cluster import KMeans
import plotly.express as px
import google.generativeai as genai
import os

# --- Configuration & Setup ---
st.set_page_config(page_title="Census Demographic Clusters", layout="wide")

# Check for secrets
if "CENSUS_API_KEY" not in st.secrets or "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing API Keys. Please configure .streamlit/secrets.toml with CENSUS_API_KEY and GOOGLE_API_KEY.")
    st.stop()

# Initialize APIs
c = Census(st.secrets["CENSUS_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# 
model = genai.GenerativeModel('gemini-2.5-flash')

# ACS 5-Year Variables (2021 is the standard stable vintage for most libraries)
# B01003_001E: Total Population
# B19013_001E: Median Household Income
# B01002_001E: Median Age
CENSUS_VARS = {
    'B01003_001E': 'population',
    'B19013_001E': 'median_income',
    'B01002_001E': 'median_age'
}

# --- Helper Functions ---
@st.cache_data
def get_census_data(state_fips):
    """Fetches county-level data for a specific state."""
    try:
        # Query the Census API
        data = c.acs5.state_county(
            fields=list(CENSUS_VARS.keys()) + ['NAME'],
            state_fips=state_fips,
            county_fips="*",
            year=2021
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Rename columns
        df.rename(columns=CENSUS_VARS, inplace=True)
        
        # Clean data (Census uses negative numbers like -666666666 for missing data)
        cols_to_numeric = ['population', 'median_income', 'median_age']
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with invalid/missing census data (negative values)
        df = df[(df['median_income'] > 0) & (df['median_age'] > 0) & (df['population'] > 0)]
        
        return df
    except Exception as e:
        st.error(f"Error fetching census data: {e}")
        return pd.DataFrame()

def generate_persona(cluster_stats):
    """Uses Gemini to generate a persona based on cluster statistics."""
    avg_income = cluster_stats['median_income']
    avg_age = cluster_stats['median_age']
    avg_pop = cluster_stats['population']
    
    prompt = f"""
    You are a data storyteller. I have identified a demographic cluster of counties with the following average statistics:
    - Average Median Household Income: ${avg_income:,.0f}
    - Average Median Age: {avg_age:.1f} years old
    - Average County Population: {avg_pop:,.0f} people

    Please provide a creative, catchy "Persona Name" (max 3-5 words) and a short "Description" (1 sentence) that captures the vibe of these counties. 
    Format the output exactly as: Name: [Name] | Description: [Description]
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Name: Cluster {cluster_stats['cluster']} | Description: AI generation failed. {e}"

# --- UI Layout ---
st.title("🇺🇸 AI-Powered Census Clustering")
st.markdown("Cluster US counties based on income, age, and population, then use **Gemini AI** to name the resulting demographic groups.")

# Sidebar Controls
with st.sidebar:
    st.header("Configuration")
    
    # State Selection
    state_names = [s.name for s in states.STATES]
    selected_state_name = st.selectbox("Select State", state_names, index=state_names.index("California"))
    selected_state = states.lookup(selected_state_name)
    
    # Clustering Parameters
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=3)
    min_pop = st.number_input("Minimum County Population (Ignore smaller counties)", value=10000, step=1000)
    
    run_btn = st.button("Run Analysis", type="primary")

# --- Main Analysis Flow ---
if run_btn and selected_state:
    with st.spinner(f"Fetching Census data for {selected_state.name}..."):
        df = get_census_data(selected_state.fips)
    
    if df.empty:
        st.warning("No data found. Please check your API key or try another state.")
    else:
        # Filter by Population
        df_filtered = df[df['population'] >= min_pop].copy()
        
        if len(df_filtered) < n_clusters:
            st.error(f"Not enough counties ({len(df_filtered)}) remaining after population filtering to form {n_clusters} clusters.")
        else:
            # Prepare data for clustering
            X = df_filtered[['median_income', 'median_age']]
            
            # Run KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_filtered['cluster'] = kmeans.fit_predict(X)
            
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            # --- 1. Graph (Plotly) ---
            with col1:
                st.subheader("Demographic Clusters")
                
                # We need a temporary map for the legend until we generate AI names
                df_filtered['cluster_label'] = "Cluster " + df_filtered['cluster'].astype(str)
                
                fig = px.scatter(
                    df_filtered,
                    x='median_age',
                    y='median_income',
                    size='population',
                    color='cluster_label',
                    hover_name='NAME',
                    title=f"Income vs. Age in {selected_state.name} (Size = Population)",
                    labels={'median_age': 'Median Age', 'median_income': 'Median Household Income'},
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- 2. AI Personas ---
            with col2:
                st.subheader("AI-Generated Personas")
                
                # Calculate cluster centers/stats
                cluster_stats = df_filtered.groupby('cluster')[['median_income', 'median_age', 'population']].mean().reset_index()
                
                persona_map = {}
                
                status_bar = st.progress(0)
                for index, row in cluster_stats.iterrows():
                    # Update progress
                    status_bar.progress((index + 1) / n_clusters)
                    
                    # Call Gemini
                    raw_text = generate_persona(row)
                    
                    # Parse basic text output
                    try:
                        name_part, desc_part = raw_text.split('|')
                        name = name_part.replace("Name:", "").strip()
                        desc = desc_part.replace("Description:", "").strip()
                    except:
                        name = f"Cluster {row['cluster']}"
                        desc = raw_text
                    
                    persona_map[f"Cluster {row['cluster']}"] = name
                    
                    # Display Card
                    with st.expander(f"🔹 {name}", expanded=True):
                        st.write(f"_{desc}_")
                        st.markdown(f"""
                        - **Avg Income:** ${row['median_income']:,.0f}
                        - **Avg Age:** {row['median_age']:.1f}
                        - **Avg Pop:** {row['population']:,.0f}
                        """)

                # Update the chart legend with new names (Rerender chart is tricky without rerun, 
                # so usually we just show the mapping or could re-plot, but list is sufficient for this scope)
