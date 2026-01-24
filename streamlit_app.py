import streamlit as st
import pandas as pd
from census import Census
from us import states
from sklearn.cluster import KMeans
import plotly.express as px
import google.generativeai as genai
import json
import time

# --- Configuration & Setup ---
st.set_page_config(page_title="Census Demographic Clusters", layout="wide")

# Check for secrets
if "CENSUS_API_KEY" not in st.secrets or "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing API Keys. Please configure .streamlit/secrets.toml with CENSUS_API_KEY and GOOGLE_API_KEY.")
    st.stop()

# Initialize APIs
c = Census(st.secrets["CENSUS_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Use a standard model name (Ensure this matches available models in your region)
model = genai.GenerativeModel('gemini-2.5-flash') 

# ACS 5-Year Variables
CENSUS_VARS = {
    'B01003_001E': 'population',
    'B19013_001E': 'median_income',
    'B01002_001E': 'median_age'
}

# --- Style Mapping ---
CLUSTER_STYLE = [
    {'color': '#636EFA', 'emoji': '🔵'}, # Blue
    {'color': '#EF553B', 'emoji': '🔴'}, # Red
    {'color': '#00CC96', 'emoji': '🟢'}, # Green
    {'color': '#AB63FA', 'emoji': '🟣'}, # Purple
    {'color': '#FFA15A', 'emoji': '🟠'}, # Orange
    {'color': '#FEFB84', 'emoji': '🟡'}, # Yellow
    {'color': '#A56F4F', 'emoji': '🟤'}, # Brown
    {'color': '#2F2F2F', 'emoji': '⚫'}, # Black
]

# --- Helper Functions ---

@st.cache_data
def get_census_data(state_fips):
    """Fetches county-level data for a specific state."""
    try:
        data = c.acs5.state_county(
            fields=list(CENSUS_VARS.keys()) + ['NAME'],
            state_fips=state_fips,
            county_fips="*",
            year=2021
        )
        df = pd.DataFrame(data)
        df.rename(columns=CENSUS_VARS, inplace=True)
        
        cols_to_numeric = ['population', 'median_income', 'median_age']
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter invalid data
        df = df[(df['median_income'] > 0) & (df['median_age'] > 0) & (df['population'] > 0)]
        return df
    except Exception as e:
        st.error(f"Error fetching census data: {e}")
        return pd.DataFrame()

# --- UPDATED AI FUNCTION: RETURNS ERROR STRING ---
@st.cache_data(show_spinner=False)
def generate_personas_batch(stats_df_json):
    """
    Sends ALL cluster statistics to Gemini in one call.
    Returns: Tuple (json_data, error_message)
    """
    time.sleep(0.5)
    
    prompt = f"""
    You are a data storyteller. I have performed KMeans clustering on US Counties.
    Here are the average statistics for each cluster (identified by 'cluster' ID):
    
    {stats_df_json}

    For EACH cluster ID provided in the data, provide:
    1. A creative, catchy "Persona Name" (3-5 words).
    2. A short "Description" (1 sentence) objectively describing the demographic.

    RETURN ONLY VALID JSON. The output format must be:
    {{
        "0": {{ "name": "Name Here", "description": "Description Here" }},
        "1": {{ "name": "Name Here", "description": "Description Here" }}
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.strip()
        
        # Handle cases where the model wraps code in markdown blocks
        if "```" in clean_text:
            clean_text = clean_text.replace("```json", "").replace("```", "")
            
        # Return Data and None for error
        return json.loads(clean_text), None
        
    except Exception as e:
        # Return empty dict and the ACTUAL error string
        return {}, str(e)

# --- UI Layout ---
st.title("🇺🇸 AI-Powered Census Clustering")
st.markdown("Cluster US counties based on income, age, and population, then use **Gemini AI** to name the resulting demographic groups.")

# Sidebar Controls
with st.sidebar:
    st.header("Configuration")
    state_names = [s.name for s in states.STATES]
    selected_state_name = st.selectbox("Select State", state_names, index=state_names.index("California"))
    selected_state = states.lookup(selected_state_name)
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=3)
    min_pop = st.number_input("Minimum County Population", value=10000, step=1000)
    run_btn = st.button("Run Analysis", type="primary")

# --- Main Analysis Flow ---
if run_btn and selected_state:
    with st.spinner(f"Fetching Census data for {selected_state.name}..."):
        df = get_census_data(selected_state.fips)
    
    if df.empty:
        st.warning("No data found.")
    else:
        df_filtered = df[df['population'] >= min_pop].copy()
        
        if len(df_filtered) < n_clusters:
            st.error(f"Not enough counties ({len(df_filtered)}) remaining to form {n_clusters} clusters.")
        else:
            # 1. Run KMeans
            X = df_filtered[['median_income', 'median_age']]
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_filtered['cluster'] = kmeans.fit_predict(X)

            # 2. Prepare Data for AI
            cluster_stats = df_filtered.groupby('cluster')[['median_income', 'median_age', 'population']].mean().reset_index()
            stats_json_str = cluster_stats.to_json(orient='records')

            # 3. Single API Call (Unpack Tuple)
            with st.spinner("Asking Gemini to analyze all clusters at once..."):
                ai_responses, error_log = generate_personas_batch(stats_json_str)

            # 4. Process Results
            cluster_metadata = {} 
            
            for index, row in cluster_stats.iterrows():
                cluster_id = str(row['cluster']) 
                style_idx = index % len(CLUSTER_STYLE)
                emoji = CLUSTER_STYLE[style_idx]['emoji']
                
                # Check AI response
                if cluster_id in ai_responses:
                    name = ai_responses[cluster_id].get('name', f"Cluster {cluster_id}")
                    desc = ai_responses[cluster_id].get('description', "Description unavailable")
                else:
                    name = f"Cluster {cluster_id}"
                    # --- FIX: Display the actual error message here ---
                    if error_log:
                        desc = f"⚠️ Error: {error_log}"
                    else:
                        desc = error_log

                cluster_metadata[row['cluster']] = {
                    'name': name,
                    'desc': desc,
                    'emoji': emoji,
                    'stats': row
                }
            
            # 5. Map AI Names back to DataFrame
            df_filtered['persona_name'] = df_filtered['cluster'].map(lambda x: cluster_metadata[x]['name'])
            df_filtered = df_filtered.sort_values(by='cluster')
            
            col1, col2 = st.columns([2, 1])
            
            # --- Graph (Plotly) ---
            with col1:
                st.subheader("Demographic Clusters")
                fig = px.scatter(
                    df_filtered,
                    x='median_age',
                    y='median_income',
                    size='population',
                    color='persona_name',
                    hover_name='NAME',
                    title=f"Income vs. Age in {selected_state.name}",
                    labels={'median_age': 'Median Age', 'median_income': 'Median Household Income'},
                    template="plotly_white",
                    color_discrete_sequence=[s['color'] for s in CLUSTER_STYLE] 
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- AI Personas ---
            with col2:
                st.subheader("AI-Generated Personas")
                for cluster_id, meta in cluster_metadata.items():
                    with st.expander(f"{meta['emoji']} {meta['name']}", expanded=True):
                        st.write(f"_{meta['desc']}_")
                        st.markdown(f"""
                        - **Avg Income:** ${meta['stats']['median_income']:,.0f}
                        - **Avg Age:** {meta['stats']['median_age']:.1f}
                        - **Avg Pop:** {meta['stats']['population']:,.0f}
                        """)
