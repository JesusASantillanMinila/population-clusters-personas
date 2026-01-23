import streamlit as st
import pandas as pd
from census import Census
from us import states
from sklearn.cluster import KMeans
import plotly.express as px
import google.generativeai as genai
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
model = genai.GenerativeModel('gemini-2.5-flash')

# ACS 5-Year Variables
CENSUS_VARS = {
    'B01003_001E': 'population',
    'B19013_001E': 'median_income',
    'B01002_001E': 'median_age'
}

# --- Style Mapping ---
# Forces strict color matching between Plotly and the Emojis
CLUSTER_STYLE = [
    {'color': '#636EFA', 'emoji': '🔵'}, # Blue
    {'color': '#EF553B', 'emoji': '🔴'}, # Red
    {'color': '#00CC96', 'emoji': '🟢'}, # Green
    {'color': '#AB63FA', 'emoji': '🟣'}, # Purple
    {'color': '#FFA15A', 'emoji': '🟠'}, # Orange
    {'color': '#FEFB84', 'emoji': '🟡'}, # Yellow
    {'color': '#A56F4F', 'emoji': '🟤'}, # Brown
    {'color': '#FF97FF', 'emoji': '🩷'}, # Pink
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

# Added caching to prevent hitting API limits on re-runs
@st.cache_data(show_spinner=False)
def generate_persona(avg_income, avg_age, avg_pop):
    """Uses Gemini to generate a persona based on cluster statistics."""
    
    # 1. Rate Limit Protection: Pause to avoid 429 errors
    time.sleep(1.5) 

    prompt = f"""
    You are a data storyteller. I have identified a demographic cluster of counties with:
    - Average Median Household Income: ${avg_income:,.0f}
    - Average Median Age: {avg_age:.1f} years old
    - Average County Population: {avg_pop:,.0f} people

    Please provide a creative, catchy "Persona Name" (max 3-5 words) and a short "Description" (1 sentence) that objectively describes these counties. 
    Format the output exactly as: Name: [Name] | Description: [Description]
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # Return a safe error string so the app doesn't crash
        return f"Name: Cluster Unknown | Description: AI Error: {str(e)}"

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

            # 2. Generate AI Personas & Prepare Metadata FIRST
            cluster_stats = df_filtered.groupby('cluster')[['median_income', 'median_age', 'population']].mean().reset_index()
            cluster_metadata = {} 
            
            # Progress bar for AI generation
            progress_text = "Generating AI personas (with rate limiting)..."
            my_bar = st.progress(0, text=progress_text)

            for index, row in cluster_stats.iterrows():
                # Assign Style based on index
                style_idx = index % len(CLUSTER_STYLE)
                emoji = CLUSTER_STYLE[style_idx]['emoji']
                
                # Call AI (Cached)
                raw_text = generate_persona(row['median_income'], row['median_age'], row['population'])
                
                # Parse Result
                try:
                    name_part, desc_part = raw_text.split('|')
                    name = name_part.replace("Name:", "").strip()
                    desc = desc_part.replace("Description:", "").strip()
                except:
                    name = f"Cluster {row['cluster']}"
                    desc = raw_text
                
                cluster_metadata[row['cluster']] = {
                    'name': name,
                    'desc': desc,
                    'emoji': emoji,
                    'stats': row
                }
                
                my_bar.progress((index + 1) / n_clusters)
            
            my_bar.empty()

            # 3. Map AI Names back to DataFrame for Plotting
            df_filtered['persona_name'] = df_filtered['cluster'].map(lambda x: cluster_metadata[x]['name'])
            
            # Sort to ensure colors align with the legend order
            df_filtered = df_filtered.sort_values(by='cluster')
            
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            # --- Graph (Plotly) ---
            with col1:
                st.subheader("Demographic Clusters")
                
                fig = px.scatter(
                    df_filtered,
                    x='median_age',
                    y='median_income',
                    size='population',
                    color='persona_name', # Use AI Name for Legend
                    hover_name='NAME',
                    title=f"Income vs. Age in {selected_state.name}",
                    labels={'median_age': 'Median Age', 'median_income': 'Median Household Income'},
                    template="plotly_white",
                    # Enforce the specific color sequence
                    color_discrete_sequence=[s['color'] for s in CLUSTER_STYLE] 
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- AI Personas (Expanders) ---
            with col2:
                st.subheader("AI-Generated Personas")
                
                for cluster_id, meta in cluster_metadata.items():
                    name = meta['name']
                    desc = meta['desc']
                    emoji = meta['emoji'] # Use Matching Emoji
                    row = meta['stats']
                    
                    with st.expander(f"{emoji} {name}", expanded=True):
                        st.write(f"_{desc}_")
                        st.markdown(f"""
                        - **Avg Income:** ${row['median_income']:,.0f}
                        - **Avg Age:** {row['median_age']:.1f}
                        - **Avg Pop:** {row['population']:,.0f}
                        """)
