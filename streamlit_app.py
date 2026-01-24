import streamlit as st
import pandas as pd
from census import Census
from us import states
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import json
import itertools

# --- Configuration & Setup ---
st.set_page_config(page_title="Demographic Persona Generator", layout="wide")

# Check secrets
if "CENSUS_API_KEY" not in st.secrets or "GOOGLE_API_KEY" not in st.secrets:
    st.error("Please add CENSUS_API_KEY and GOOGLE_API_KEY to your .streamlit/secrets.toml file.")
    st.stop()

# Initialize APIs
c = Census(st.secrets["CENSUS_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- Session State Initialization ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None
if 'summary_df' not in st.session_state:
    st.session_state.summary_df = None

# --- Census Variables Mapping ---
CENSUS_VARS = {
    'B01002_001E': 'Median Age',
    'B19013_001E': 'Median Income',
    'B01003_001E': 'Population',
    # Education
    'B15003_001E': 'Edu_Total',
    'B15003_022E': 'Edu_Bach',
    'B15003_023E': 'Edu_Mast',
    'B15003_024E': 'Edu_Prof',
    'B15003_025E': 'Edu_Doc',
    # Housing Tenure
    'B25003_001E': 'Housing_Total',
    'B25003_002E': 'Housing_Owner',
    # Household Structure
    'B11001_001E': 'HH_Total',
    'B11001_002E': 'HH_Family'
}

# --- Helper Functions ---

@st.cache_data
def get_census_data(state_fips):
    """Fetches county-level data and calculates derived demographic percentages."""
    try:
        data = c.acs5.state_county(
            fields=list(CENSUS_VARS.keys()) + ['NAME'],
            state_fips=state_fips,
            county_fips="*"
        )
        df = pd.DataFrame(data)
        
        # Rename columns based on mapping
        df = df.rename(columns=CENSUS_VARS)
        
        # Convert to numeric
        numeric_cols = list(CENSUS_VARS.values())
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # --- Feature Engineering ---
        # 1. Education: % Bachelor's Degree or Higher
        df['Edu_Sum_Higher'] = df['Edu_Bach'] + df['Edu_Mast'] + df['Edu_Prof'] + df['Edu_Doc']
        df['Pct Bachelor+'] = (df['Edu_Sum_Higher'] / df['Edu_Total']) * 100
        
        # 2. Housing: % Owner Occupied
        df['Pct Owner Occupied'] = (df['Housing_Owner'] / df['Housing_Total']) * 100
        
        # 3. Structure: % Family Households
        df['Pct Family Households'] = (df['HH_Family'] / df['HH_Total']) * 100
        
        # Clean up infinite values or NaNs from division by zero
        final_cols = ['NAME', 'Population', 'Median Age', 'Median Income', 
                      'Pct Bachelor+', 'Pct Owner Occupied', 'Pct Family Households']
        
        df = df[final_cols].dropna()
        
        return df
    except Exception as e:
        st.error(f"Error fetching Census data: {e}")
        return pd.DataFrame()

def generate_personas(cluster_summary):
    """Generates persona names using Gemini based on 5 demographic variables."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Create a string representation of the clusters
    cluster_text = ""
    for index, row in cluster_summary.iterrows():
        cluster_text += (
            f"Cluster {index}: "
            f"Age {row['Median Age']:.1f}, "
            f"Income ${row['Median Income']:,.0f}, "
            f"Education (Bach+) {row['Pct Bachelor+']:.1f}%, "
            f"Home Ownership {row['Pct Owner Occupied']:.1f}%, "
            f"Family Households {row['Pct Family Households']:.1f}%\n"
        )

    prompt = f"""
    You are a demographic analyst. I have clustered US counties based on Age, Income, Education, Housing Tenure, and Household Structure.
    Here are the statistics for each cluster:
    
    {cluster_text}
    
    For EACH cluster, provide:
    1. A creative, short "Persona Name" (e.g., "Educated Suburban Families", "Retiring Rural Owners").
    2. A short objective description (max 2 sentences) explaining *why* they fit this name based on the specific variables provided.
    
    Return the response strictly as a JSON list of objects with keys: "cluster_id", "persona_name", "description". 
    Do not include markdown formatting like ```json. Just the raw JSON string.
    """
    
    try:
        response = model.generate_content(prompt)
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text_response)
    except Exception as e:
        st.error(f"Error generating personas with Gemini: {e}")
        return []

# --- Main App Interface ---

st.title("🇺🇸 Deep Demographic Clustering")
st.markdown("Analyze US counties by Age, Income, Education, Housing & Family Structure.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    
    state_list = [s.name for s in states.STATES]
    selected_state_name = st.selectbox("Select State", state_list, index=state_list.index("California"))
    selected_state = states.lookup(selected_state_name)     
    
    min_pop = st.number_input("Minimum County Population", min_value=0, value=10000, step=1000)
    num_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=4)
    
    # Logic: If inputs change, we might want to reset the "loaded" state, 
    # but for now, we rely on the button to force a new run.
    run_btn = st.button("Run Analysis", type="primary")

# --- Processing Block (Only runs when button clicked) ---
if run_btn:
    with st.spinner(f"Fetching Census data for {selected_state_name}..."):
        df = get_census_data(selected_state.fips)

    if not df.empty:
        df_filtered = df[df['Population'] >= min_pop].copy()
        
        if len(df_filtered) < num_clusters:
            st.error(f"Not enough counties remain after filtering (Count: {len(df_filtered)}). Decrease the population threshold.")
            st.session_state.data_loaded = False
        else:
            # --- Machine Learning Step ---
            features = ['Median Age', 'Median Income', 'Pct Bachelor+', 'Pct Owner Occupied', 'Pct Family Households']
            X = df_filtered[features]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Calculate Cluster Centers
            centers_scaled = kmeans.cluster_centers_
            centers = scaler.inverse_transform(centers_scaled)
            
            summary_df = pd.DataFrame(centers, columns=features)
            summary_df['Counties in Cluster'] = df_filtered['Cluster'].value_counts().sort_index()
            
            # --- GenAI Step ---
            with st.spinner("Generating detailed personas with Gemini..."):
                personas = generate_personas(summary_df)
            
            # Process GenAI Results
            if personas:
                persona_map = {p['cluster_id']: p for p in personas}
                
                summary_df['Persona Name'] = summary_df.index.map(lambda x: persona_map.get(x, {}).get('persona_name', 'Unknown'))
                summary_df['Description'] = summary_df.index.map(lambda x: persona_map.get(x, {}).get('description', 'No description'))
                
                df_filtered['Persona Name'] = df_filtered['Cluster'].map(lambda x: persona_map.get(x, {}).get('persona_name', f'Cluster {x}'))
            
            # --- Save to Session State ---
            st.session_state.df_filtered = df_filtered
            st.session_state.summary_df = summary_df
            st.session_state.data_loaded = True

    else:
        st.warning("No data found. Please check your API key or try a different state.")
        st.session_state.data_loaded = False

# --- Visualization Block (Runs if data is loaded) ---
if st.session_state.data_loaded:
    
    # Retrieve data from session state
    df_filtered = st.session_state.df_filtered
    summary_df = st.session_state.summary_df
    features = ['Median Age', 'Median Income', 'Pct Bachelor+', 'Pct Owner Occupied', 'Pct Family Households']
    
    st.success("Analysis Complete!")
    
    # 1. Visualization Controls (Cycle through combinations)
    st.subheader("Visual Analysis")
    
    # Generate all unique pairs of features for the graph options
    feature_pairs = list(itertools.combinations(features, 2))
    graph_options = {f"{x} vs {y}": (x, y) for x, y in feature_pairs}
    
    # User Selector
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Graph Filter**")
        selected_graph_label = st.radio(
            "Choose Graph Combination:", 
            list(graph_options.keys()),
            index=0 
        )
        x_axis, y_axis = graph_options[selected_graph_label]
    
    with col2:
        fig = px.scatter(
            df_filtered,
            x=x_axis,
            y=y_axis,
            color='Persona Name',
            hover_data=['NAME', 'Population'] + features,
            size='Population',
            title=f"{selected_graph_label} (Colored by Persona)",
            template="plotly_white",
            height=500
        )
        fig.update_layout(legend_title_text='AI Persona')
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Detailed Summary Table
    st.divider()
    st.subheader("Cluster Personas & Statistics")
    
    # Formatting for display
    summary_display = summary_df.copy()
    summary_display['Median Income'] = summary_display['Median Income'].apply(lambda x: f"${x:,.0f}")
    summary_display['Median Age'] = summary_display['Median Age'].apply(lambda x: f"{x:.1f}")
    summary_display['Pct Bachelor+'] = summary_display['Pct Bachelor+'].apply(lambda x: f"{x:.1f}%")
    summary_display['Pct Owner Occupied'] = summary_display['Pct Owner Occupied'].apply(lambda x: f"{x:.1f}%")
    summary_display['Pct Family Households'] = summary_display['Pct Family Households'].apply(lambda x: f"{x:.1f}%")
    
    cols = ['Persona Name', 'Median Age', 'Median Income', 'Pct Bachelor+', 'Pct Owner Occupied', 'Pct Family Households', 'Counties in Cluster', 'Description']
    
    st.dataframe(
        summary_display[cols], 
        hide_index=True, 
        use_container_width=True
    )
