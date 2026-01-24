import streamlit as st
import pandas as pd
from census import Census
from us import states
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import json

# --- Configuration & Setup ---
st.set_page_config(page_title="Demographic Persona Generator", layout="wide")

# check secrets
if "CENSUS_API_KEY" not in st.secrets or "GOOGLE_API_KEY" not in st.secrets:
    st.error("Please add CENSUS_API_KEY and GOOGLE_API_KEY to your .streamlit/secrets.toml file.")
    st.stop()

# Initialize APIs
c = Census(st.secrets["CENSUS_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Mapping of Census Variables
# B01002_001E: Median Age
# B19013_001E: Median Household Income
# B01003_001E: Total Population
CENSUS_VARS = {
    'B01002_001E': 'Median Age',
    'B19013_001E': 'Median Income',
    'B01003_001E': 'Population'
}

# --- Helper Functions ---

@st.cache_data
def get_census_data(state_fips):
    """Fetches county-level data for a specific state."""
    try:
        data = c.acs5.state_county(
            fields=list(CENSUS_VARS.keys()) + ['NAME'],
            state_fips=state_fips,
            county_fips="*"
        )
        df = pd.DataFrame(data)
        
        # Rename columns
        df = df.rename(columns=CENSUS_VARS)
        
        # Convert numeric columns to float/int
        numeric_cols = ['Median Age', 'Median Income', 'Population']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Drop rows with missing data (often occurs in very small counties)
        df = df.dropna(subset=numeric_cols)
        
        return df
    except Exception as e:
        st.error(f"Error fetching Census data: {e}")
        return pd.DataFrame()

def generate_personas(cluster_summary):
    """

    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Create a string representation of the clusters for the prompt
    cluster_text = ""
    for index, row in cluster_summary.iterrows():
        cluster_text += f"Cluster {index}: Median Age {row['Median Age']:.1f}, Median Income ${row['Median Income']:,.0f}\n"

    prompt = f"""
    You are a demographic analyst. I have clustered US counties based on age and income. 
    Here are the statistics for each cluster:
    
    {cluster_text}
    
    For EACH cluster, provide:
    1. A creative, short "Persona Name" (e.g., "Wealthy Retirees", "Young Professionals").
    2. A short objective description (max 3 sentences).
    
    Return the response strictly as a JSON list of objects with keys: "cluster_id", "persona_name", "description". 
    Do not include markdown formatting like ```json. Just the raw JSON string.
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean response just in case markdown was included
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text_response)
    except Exception as e:
        st.error(f"Error generating personas with Gemini: {e}")
        return []

# --- Main App Interface ---

st.title("🇺🇸 AI-Powered Demographic Clustering")
st.markdown("Analyze US counties by Age and Income, clustered by ML and named by GenAI.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    
    # State Selection
    state_list = [s.name for s in states.STATES]
    selected_state_name = st.selectbox("Select State", state_list, index=state_list.index("California"))
    
    # Use the .lookup() method instead of getattr
    selected_state = states.lookup(selected_state_name)    
    # Filter Controls
    min_pop = st.number_input("Minimum County Population", min_value=0, value=10000, step=1000)
    
    # Clustering Controls
    num_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=4)
    
    run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    with st.spinner(f"Fetching Census data for {selected_state_name}..."):
        df = get_census_data(selected_state.fips)

    if not df.empty:
        # Filter by Population
        df_filtered = df[df['Population'] >= min_pop].copy()
        
        if len(df_filtered) < num_clusters:
            st.error(f"Not enough counties remain after filtering (Count: {len(df_filtered)}). Decrease the population threshold.")
        else:
            # --- Machine Learning Step ---
            features = ['Median Age', 'Median Income']
            X = df_filtered[features]
            
            # Standardize Data (Important for K-Means)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # KMeans Clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Calculate Cluster Centers (inverse transform to get real values)
            centers_scaled = kmeans.cluster_centers_
            centers = scaler.inverse_transform(centers_scaled)
            
            # Create Summary DataFrame
            summary_df = pd.DataFrame(centers, columns=features)
            summary_df['Counties in Cluster'] = df_filtered['Cluster'].value_counts().sort_index()
            
            # --- GenAI Step ---
            with st.spinner("Generating personas with Gemini 2.5 Flash..."):
                personas = generate_personas(summary_df)
            
            # Process GenAI Results
            if personas:
                # Create a map from cluster_id to persona data
                persona_map = {p['cluster_id']: p for p in personas}
                
                # Add GenAI data to summary df
                summary_df['Persona Name'] = summary_df.index.map(lambda x: persona_map.get(x, {}).get('persona_name', 'Unknown'))
                summary_df['Description'] = summary_df.index.map(lambda x: persona_map.get(x, {}).get('description', 'No description'))
                
                # Add Persona Name to main dataframe for plotting
                df_filtered['Persona Name'] = df_filtered['Cluster'].map(lambda x: persona_map.get(x, {}).get('persona_name', f'Cluster {x}'))
            
            # --- Visualizations ---
            
            # 1. Plotly Graph
            st.subheader(f"Demographic Clusters in {selected_state_name}")
            fig = px.scatter(
                df_filtered,
                x='Median Age',
                y='Median Income',
                color='Persona Name',
                hover_data=['NAME', 'Population'],
                size='Population',
                title=f"Income vs Age by County (Colored by AI Persona)",
                template="plotly_white"
            )
            fig.update_layout(legend_title_text='AI Persona')
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. Summary Table
            st.subheader("Cluster Personas & Statistics")
            
            # Reorder columns for display
            display_cols = ['Persona Name', 'Median Age', 'Median Income', 'Counties in Cluster', 'Description']
            
            # Format numbers for cleaner display
            summary_display = summary_df.copy()
            summary_display['Median Income'] = summary_display['Median Income'].apply(lambda x: f"${x:,.0f}")
            summary_display['Median Age'] = summary_display['Median Age'].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(
                summary_display[display_cols], 
                hide_index=True, 
                use_container_width=True
            )

    else:
        st.warning("No data found. Please check your API key or try a different state.")
