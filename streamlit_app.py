import streamlit as st
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import itertools
import google.generativeai as genai

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
        # Ensure FIPS are strings and have leading zeros (e.g., 1 -> '01')
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

def generate_persona(row):
    """Uses Gemini to create a persona name and description based on cluster data."""
    prompt = f"""
    Based on the following demographic data of a population cluster:
    - Average Income: {row['Avg Income ($)']}
    - Average Age: {row['Avg Age']}
    - Most Common Education: {row['Most Common Education']}
    - Most Common Household: {row['Most Common Household']}
     
    Create a short, catchy 'Persona Name' (2-4 words) and a brief 2-sentence description of their lifestyle.
    Format your response exactly like this:
    Name: [Name Here]
    Description: [Description Here]
    """
    try:
        response = model.generate_content(prompt)
        text = response.text
        # Parsing basic response
        name = text.split("Name:")[1].split("Description:")[0].strip()
        desc = text.split("Description:")[1].strip()
        return name, desc
    except Exception as e:
        return f"Cluster {row['Cluster']}", "No description available."

@st.cache_data
def fetch_pums_data(state_fips, api_key):
    # Added PUMA to the get request
    base_url = "https://api.census.gov/data/2022/acs/acs1/pums"
    params = {"get": "AGEP,PINCP,HHT,SCHL,PUMA", "for": f"state:{state_fips}", "key": api_key}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        # PUMA remains a string to preserve leading zeros
        cols = ['AGEP', 'PINCP', 'HHT', 'SCHL']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_puma_locations(state_fips):
    """Fetches approximate coordinates for PUMAs using Census Gazetteers."""
    url = f"https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2022_Gazetteer/2022_gaz_pumas_{state_fips}.txt"
    try:
        # PUMA gazetteer files are tab-separated
        df_coords = pd.read_csv(url, sep='\t', dtype={'GEOID': str})
        df_coords.columns = df_coords.columns.str.strip()
        # Extract PUMA from GEOID (last 5 digits)
        df_coords['PUMA'] = df_coords['GEOID'].str[-5:]
        return df_coords[['PUMA', 'INTPTLAT', 'INTPTLON']]
    except Exception as e:
        st.error(f"Could not load geographic data: {e}")
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

# Check for API Key before rendering sidebar controls
if "CENSUS_API_KEY" in st.secrets:
    api_key = st.secrets["CENSUS_API_KEY"]
else:
    st.error("CENSUS_API_KEY not found in secrets.")
    st.stop()

# --- Sidebar converted to Expander ---
with st.expander("Configuration", expanded=True):
  
    st.markdown("""
    **Instructions:**
    1. Select a **US State** from the dropdown.
    2. Choose the **Number of Clusters** (groups) to identify.
    3. Click **Execute** to analyze demographics and generate AI personas.
    """)
    # --------------------------------
    
    col1, col2 = st.columns(2)
    with col1:
        selected_state = st.selectbox("Select US State", list(STATE_FIPS.keys()))
    with col2:
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
            
            # --- Obtain Geographical Data ---
            geo_df = get_puma_locations(fips)
            if not geo_df.empty:
                df = df.merge(geo_df, on='PUMA', how='left')
                df['INTPTLAT'] = pd.to_numeric(df['INTPTLAT'])
                df['INTPTLON'] = pd.to_numeric(df['INTPTLON'])

            features = ['AGEP', 'PINCP', 'HHT', 'SCHL']
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[features])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(scaled_features)
            
            # --- Generate Personas via Gemini ---
            summary_stats = df.groupby('Cluster').agg({
                'PINCP': 'mean',
                'AGEP': 'mean',
                'Education Level': lambda x: x.mode()[0] if not x.mode().empty else "N/A",
                'Household Type': lambda x: x.mode()[0] if not x.mode().empty else "N/A"
            }).reset_index()
            
            summary_stats = summary_stats.rename(columns={'PINCP': 'Avg Income ($)', 'AGEP': 'Avg Age', 'Education Level': 'Most Common Education', 'Household Type': 'Most Common Household'})
            
            persona_dict = {}
            for _, row in summary_stats.iterrows():
                name, desc = generate_persona(row)
                persona_dict[int(row['Cluster'])] = {"name": name, "desc": desc}
            
            st.session_state['persona_map'] = persona_dict
            st.session_state['data'] = df
            st.success(f"Clusters analyzed and AI personas generated!")
        else:
            st.warning("No data found.")

if st.session_state['data'] is not None:
    df = st.session_state['data']
    persona_map = st.session_state['persona_map']
    
    # Map persona names back to the main dataframe for plotting
    df['Persona Name'] = df['Cluster'].map(lambda x: persona_map[x]['name'])
    
    st.divider()
    st.subheader("📊 Cluster Visualizations")
    
    # --- New Map Section ---
    st.write("### Geographic Persona Distribution")
    map_df = df.sample(min(3000, len(df))).copy()
    if 'INTPTLAT' in map_df.columns:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="INTPTLAT",
            lon="INTPTLON",
            color="Persona Name",
            size_max=15,
            zoom=5,
            mapbox_style="carto-positron",
            title=f"Geographic Clustering in {selected_state}",
            hover_data=['Avg Age' if 'Avg Age' in map_df.columns else 'AGEP']
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Geographic coordinates not available for this state.")

    st.divider()
    
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
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Summary Table ---
    st.divider()
    st.subheader("📋 Persona Summaries")
    
    final_summary = df.groupby('Cluster').agg({
        'PINCP': 'mean',
        'AGEP': 'mean',
        'Education Level': lambda x: x.mode()[0],
        'Household Type': lambda x: x.mode()[0]
    }).reset_index()

    # Add AI Persona Name and Description to the final summary
    final_summary['Persona Name'] = final_summary['Cluster'].map(lambda x: persona_map[x]['name'])
    final_summary['Persona Description'] = final_summary['Cluster'].map(lambda x: persona_map[x]['desc'])
    
    # Clean up formatting
    final_summary['Avg Income ($)'] = final_summary['PINCP'].round().astype(int).apply(lambda x: f"${x:,}")
    final_summary['Avg Age'] = final_summary['AGEP'].round().astype(int)
    
    # Reorder columns for readability
    display_cols = ['Persona Name', 'Persona Description', 'Avg Income ($)', 'Avg Age', 'Education Level', 'Household Type']
    
    # We use .style.hide() to remove the index, and set formatting specific to the table view
    st.table(
        final_summary[display_cols].style.hide(axis="index")
    )

elif st.session_state['data'] is None:
    pass
