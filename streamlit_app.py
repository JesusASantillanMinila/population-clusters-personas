import streamlit as st
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import itertools
import google.generativeai as genai
import json
import re

# --- 1. Configuration & Dictionaries ---
st.set_page_config(page_title="US Population Personas Clustering", layout="wide")

# Configure Gemini
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash') 
else:
    st.error("GOOGLE_API_KEY not found in secrets.")

# Load State FIPS from CSV
@st.cache_data
def load_fips_data():
    try:
        df = pd.read_csv('state_fips.csv')
        df['FIPS'] = df['FIPS'].astype(str).str.zfill(2)
        return dict(zip(df['State'], df['FIPS']))
    except FileNotFoundError:
        st.error("state_fips.csv not found. Please ensure the file exists.")
        return {}

STATE_FIPS = load_fips_data()

EDUCATION_MAP = {
    '16': 'High School Diploma', '17': 'GED', '18': 'Some College (<1yr)',
    '19': 'Some College (1+yr)', '20': 'Associate Degree', 
    '21': 'Bachelor\'s Degree', '22': 'Master\'s Degree', 
    '23': 'Professional Degree', '24': 'Doctorate'
}

EDU_ORDER = list(EDUCATION_MAP.values()) + ['Other/Less than HS']

HHT_MAP = {
    '1': 'Married Couple', '2': 'Male Householder (No Spouse)',
    '3': 'Female Householder (No Spouse)', '4': 'Non-Family (Male)',
    '5': 'Non-Family (Female)', '6': 'Group Quarters', '7': 'Group Quarters'
}

# --- 2. Helper Functions ---

def generate_personas_batch(summary_stats_df):
    """
    Combines all cluster data into one prompt to use only 1 Gemini API call.
    Includes error handling for API quota limits.
    """
    cluster_info = summary_stats_df.to_string(index=False)
    
    prompt = f"""
    Based on the following demographic data for multiple population clusters:
    {cluster_info}
    
    For EACH Cluster ID, create a short, catchy 'Persona Name' (2-4 words) and a brief 2-sentence description of their lifestyle.
    
    Return the response ONLY as a JSON object where the keys are the Cluster IDs.
    Example Format:
    {{
      "0": {{"name": "Name Here", "desc": "Description Here"}},
      "1": {{"name": "Name Here", "desc": "Description Here"}}
    }}
    """
    try:
        response = model.generate_content(prompt)
        json_text = re.sub(r"```json|```", "", response.text).strip()
        return json.loads(json_text)
        
    except Exception as e:
        # Check if the error is related to quota/rate limits
        error_msg = str(e).lower()
        if "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg:
            st.error("⚠️ **AI Token Limit Reached:** You have exhausted the allocated AI tokens for the current 24-hour period. Please wait and try rerunning the model later.")
        else:
            st.error(f"Batch generation failed: {e}")
            
        # Fallback to generic names so the app doesn't crash
        return {str(i): {"name": f"Cluster {i}", "desc": "AI description unavailable due to limit."} for i in summary_stats_df['Cluster']}

@st.cache_data
def fetch_pums_data(state_fips, api_key):
    base_url = "https://api.census.gov/data/2022/acs/acs1/pums"
    params = {"get": "AGEP,PINCP,HHT,SCHL,PUMA", "for": f"state:{state_fips}", "key": api_key}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        cols = ['AGEP', 'PINCP', 'HHT', 'SCHL', 'PUMA']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def process_data(df):
    df = df[(df['PINCP'] >= 0) & (df['AGEP'] > 18)].copy()
    bins = [18, 30, 45, 60, 100]
    labels = ['18-30', '31-45', '46-60', '60+']
    df['Age Range'] = pd.cut(df['AGEP'], bins=bins, labels=labels)
    df['Education Level'] = df['SCHL'].astype(str).map(EDUCATION_MAP).fillna('Other/Less than HS')
    df['Household Type'] = df['HHT'].astype(str).map(HHT_MAP).fillna('Unknown')
    return df

# --- 3. Main Streamlit App ---

st.title("👥 US Population Segmenter")

if "CENSUS_API_KEY" in st.secrets:
    api_key = st.secrets["CENSUS_API_KEY"]
else:
    st.error("CENSUS_API_KEY not found in secrets.")
    st.stop()

with st.expander("Configuration", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Instructions:**
        1. Select a **US State** from the dropdown.
        2. Choose the **Number of Clusters** (groups) to identify.
        3. Click **Execute** to analyze demographics and generate AI personas.
        """)
    with col2:
        selected_state = st.selectbox("Select US State", list(STATE_FIPS.keys()))
        n_clusters = st.slider("Number of Clusters (k)", 2, 8, 3)
        
    execute_btn = st.button("Execute")

if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'persona_map' not in st.session_state:
    st.session_state['persona_map'] = {}

if execute_btn:
    with st.spinner(f"Analyzing {selected_state} demographics with AI..."):
        fips = STATE_FIPS[selected_state]
        raw_df = fetch_pums_data(fips, api_key)
        
        if not raw_df.empty:
            df = process_data(raw_df)
            features = ['AGEP', 'PINCP', 'HHT', 'SCHL']
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[features])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(scaled_features)
            
            summary_stats = df.groupby('Cluster').agg({
                'PINCP': 'mean',
                'AGEP': 'mean',
                'Education Level': lambda x: x.mode()[0] if not x.mode().empty else "N/A",
                'Household Type': lambda x: x.mode()[0] if not x.mode().empty else "N/A"
            }).reset_index()
            
            summary_stats = summary_stats.rename(columns={'PINCP': 'Avg Income ($)', 'AGEP': 'Avg Age', 'Education Level': 'Most Common Education', 'Household Type': 'Most Common Household'})
            
            batch_results = generate_personas_batch(summary_stats)
            
            persona_dict = {}
            for cluster_id, content in batch_results.items():
                persona_dict[int(cluster_id)] = {"name": content['name'], "desc": content['desc']}
            
            st.session_state['persona_map'] = persona_dict
            st.session_state['data'] = df
            st.success(f"Analysis Finalized")
        else:
            st.warning("No data found.")

if st.session_state['data'] is not None:
    df = st.session_state['data']
    persona_map = st.session_state['persona_map']
    df['Persona Name'] = df['Cluster'].map(lambda x: persona_map[x]['name'])
    
    st.divider()
    st.subheader("📊 Cluster Visualizations")
    
    plot_vars = {'Age': 'AGEP', 'Income': 'PINCP', 'Education': 'Education Level', 'Household Type': 'Household Type'}
    combinations = list(itertools.combinations(plot_vars.keys(), 2))
    selected_combo = st.selectbox("Select Graph to Show:", [f"{x} vs {y}" for x, y in combinations])
    x_label, y_label = next((x, y) for x, y in combinations if f"{x} vs {y}" == selected_combo)
    
    plot_df = df.sample(min(2000, len(df))).copy()
    
    fig = px.scatter(
        plot_df, 
        x=plot_vars[x_label], 
        y=plot_vars[y_label],
        color='Persona Name', 
        title=f"Persona Distribution: {x_label} vs {y_label}",
        hover_data=['Education Level', 'Household Type'],
        labels={'AGEP': 'Age', 'PINCP': 'Annual Income', 'Persona Name': 'Market Persona'},
        category_orders={'Education Level': EDU_ORDER}
    )
    
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)

    # --- Map Visualization Section ---
    st.divider()
    st.subheader("🗺️ Geographic Persona Distribution")
    try:
        geo_df = pd.read_csv('puma_coordinates.csv')
        geo_df['state'] = geo_df['state'].astype(int)
        geo_df['PUMA'] = geo_df['PUMA'].astype(int)
        
        current_fips = int(STATE_FIPS[selected_state])
        state_geo = geo_df[geo_df['state'] == current_fips]
        
        map_df = df.sample(min(3000, len(df))).copy()
        map_df = map_df.merge(state_geo[['PUMA', 'IntPtLat', 'IntPtLon', 'pumaName']], on='PUMA', how='left')
        map_df = map_df.dropna(subset=['IntPtLat', 'IntPtLon'])

        if not map_df.empty:
            fig_map = px.scatter_mapbox(
                map_df,
                lat="IntPtLat",
                lon="IntPtLon",
                color="Persona Name",
                hover_name="pumaName",
                hover_data={"IntPtLat": False, "IntPtLon": False, "Persona Name": True, "Age Range": True},
                zoom=5,
                height=600,
                title=f"Geographic Distribution of Personas in {selected_state}"
            )
            fig_map.update_layout(mapbox_style="carto-positron")
            # Update map legend to be underneath
            fig_map.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Could not find matching PUMA coordinates for this state in puma_coordinates.csv.")
    except FileNotFoundError:
        st.error("puma_coordinates.csv not found. Please upload it to use the mapping feature.")
    except Exception as e:
        st.error(f"Error generating map: {e}")

    # --- Summary Section (Updated to 3-Column Container Layout) ---
    st.divider()
    st.subheader("📋 Persona Summaries")
    
    # Calculate percentages
    cluster_counts = df['Cluster'].value_counts(normalize=True) * 100
    
    summary_data = df.groupby('Cluster').agg({
        'PINCP': 'mean',
        'AGEP': 'mean',
        'Education Level': lambda x: x.mode()[0],
        'Household Type': lambda x: x.mode()[0]
    }).reset_index()

    persona_cols = st.columns(3)
    for idx, row in summary_data.iterrows():
        p_id = int(row['Cluster'])
        col_idx = idx % 3
        
        income_k = f"${int(row['PINCP'] / 1000)}K"
        cluster_pct = f"{cluster_counts[p_id]:.1f}%"
        
        with persona_cols[col_idx]:
            with st.container(border=True):
                st.markdown(f"### {persona_map[p_id]['name']}")
                st.caption(persona_map[p_id]['desc'])
                st.write("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Income", income_k)
                c2.metric("Avg Age", f"{int(row['AGEP'])}")
                c3.metric("% of Pop.", cluster_pct)
                st.markdown(f"**Top Education:** {row['Education Level']}")
                st.markdown(f"**Top Household:** {row['Household Type']}")
elif st.session_state['data'] is None:
    pass
