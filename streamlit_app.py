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
import numpy as np

# --- 1. Configuration & Dictionaries ---
st.set_page_config(page_title="US Population Personas Clustering", layout="wide")

# Configure Gemini
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash') 
else:
    st.error("GOOGLE_API_KEY not found in secrets.")

# Load State FIPS from CSV instead of hardcoding
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
        st.error(f"Batch generation failed: {e}")
        return {str(i): {"name": f"Cluster {i}", "desc": "No description available."} for i in summary_stats_df['Cluster']}

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

st.title("US Population Personas Clustering")

if "CENSUS_API_KEY" in st.secrets:
    api_key = st.secrets["CENSUS_API_KEY"]
else:
    st.error("CENSUS_API_KEY not found in secrets.")
    st.stop()

# Initialize session state
if 'elbow_fig' not in st.session_state:
    st.session_state['elbow_fig'] = None
if 'suggested_k' not in st.session_state:
    st.session_state['suggested_k'] = 3

def on_state_change():
    """Triggered when the state selection changes to calculate elbow immediately."""
    with st.spinner(f"Calculating optimal clusters for {st.session_state.selected_state}..."):
        fips = STATE_FIPS[st.session_state.selected_state]
        raw_df = fetch_pums_data(fips, api_key)
        if not raw_df.empty:
            df = process_data(raw_df)
            features = ['AGEP', 'PINCP', 'HHT', 'SCHL']
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[features])
            
            wcss = []
            k_range = range(2, 9)
            for i in k_range:
                km = KMeans(n_clusters=i, random_state=42, n_init=10)
                km.fit(scaled_features)
                wcss.append(km.inertia_)
            
            # Simple Knee Detection: Point with max distance from line connecting start and end
            x = np.array(list(k_range))
            y = np.array(wcss)
            line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
            line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
            vec_from_first = np.column_stack([x - x[0], y - y[0]])
            scalar_prod = np.sum(vec_from_first * line_vec_norm, axis=1)
            vec_to_line = vec_from_first - np.outer(scalar_prod, line_vec_norm)
            dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
            st.session_state['suggested_k'] = int(x[np.argmax(dist_to_line)])

            fig_elbow = px.line(x=list(k_range), y=wcss, markers=True, 
                                title=f"Elbow Method for {st.session_state.selected_state}",
                                labels={'x': 'Number of Clusters (k)', 'y': 'WCSS'})
            fig_elbow.add_vline(x=st.session_state['suggested_k'], line_dash="dash", line_color="green", annotation_text="Suggested k")
            st.session_state['elbow_fig'] = fig_elbow

with st.expander("Configuration", expanded=True):
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("""
        **Instructions:**
        1. Select a **US State**.
        2. View the **Elbow Graph** to determine optimal $k$.
        3. Choose the **Number of Clusters** and **Execute**.
        """)
        selected_state = st.selectbox("Select US State", list(STATE_FIPS.keys()), key="selected_state", on_change=on_state_change)
        
        # If the app just loaded and no elbow graph exists, run once for default state
        if st.session_state['elbow_fig'] is None:
            on_state_change()
            st.rerun()

        n_clusters = st.slider("Number of Clusters (k)", 2, 8, st.session_state['suggested_k'])
        execute_btn = st.button("Execute Analysis", use_container_width=True)

    with col2:
        if st.session_state['elbow_fig']:
            st.plotly_chart(st.session_state['elbow_fig'], use_container_width=True)
            st.success(f"💡 **Suggestion:** Based on the elbow curve, **k={st.session_state['suggested_k']}** appears optimal.")
        else:
            st.info("Select a state to see the Elbow Graph.")

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
            persona_dict = {int(k): v for k, v in batch_results.items()}
            
            st.session_state['persona_map'] = persona_dict
            st.session_state['data'] = df
            st.rerun()
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
        plot_df, x=plot_vars[x_label], y=plot_vars[y_label],
        color='Persona Name', title=f"Persona Distribution: {x_label} vs {y_label}",
        hover_data=['Education Level', 'Household Type'],
        labels={'AGEP': 'Age', 'PINCP': 'Annual Income', 'Persona Name': 'Market Persona'},
        category_orders={'Education Level': EDU_ORDER}
    )
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
                map_df, lat="IntPtLat", lon="IntPtLon", color="Persona Name",
                hover_name="pumaName", zoom=5, height=600,
                title=f"Geographic Distribution of Personas in {selected_state}"
            )
            fig_map.update_layout(mapbox_style="carto-positron")
            st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.info("Note: Map view requires a valid puma_coordinates.csv file.")

    # --- 2) Attractive Persona Summaries ---
    st.divider()
    st.subheader("📋 Persona Detailed Profiles")
    
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
        with persona_cols[col_idx]:
            with st.container(border=True):
                st.markdown(f"### {persona_map[p_id]['name']}")
                st.caption(persona_map[p_id]['desc'])
                st.write("---")
                c1, c2 = st.columns(2)
                c1.metric("Avg Income", f"${int(row['PINCP']):,}")
                c2.metric("Avg Age", f"{int(row['AGEP'])}")
                st.markdown(f"**Top Education:** {row['Education Level']}")
                st.markdown(f"**Top Household:** {row['Household Type']}")

elif st.session_state['data'] is None:
    pass
