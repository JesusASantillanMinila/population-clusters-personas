import streamlit as st
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import itertools 
import google.generativeai as genai
import json

# --- 1. Configuration & Dictionaries ---
st.set_page_config(page_title="US Population Personas Clustering", layout="wide")

# Configure Gemini
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.sidebar.error("GOOGLE_API_KEY not found in secrets.")

STATE_FIPS = {
    "Alabama": "01", "California": "06", "Colorado": "08", 
    "Florida": "12", "Georgia": "13", "Illinois": "17", 
    "New York": "36", "Texas": "48", "Virginia": "51", "Washington": "53"
}

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

@st.cache_data
def fetch_pums_data(state_fips, api_key):
    base_url = "https://api.census.gov/data/2022/acs/acs1/pums"
    params = {"get": "AGEP,PINCP,HHT,SCHL"
              , "for": f"state:{state_fips}"
              , "key": api_key}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        for c in ['AGEP', 'PINCP', 'HHT', 'SCHL']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def process_data(df):
    df = df[(df['PINCP'] >= 0) & (df['AGEP'] > 18)]
    bins = [18, 30, 45, 60, 100]
    labels = ['18-30', '31-45', '46-60', '60+']
    df['Age Range'] = pd.cut(df['AGEP'], bins=bins, labels=labels)
    df['Education Level'] = df['SCHL'].astype(str).map(EDUCATION_MAP).fillna('Other/Less than HS')
    df['Household Type'] = df['HHT'].astype(str).map(HHT_MAP).fillna('Unknown')
    return df

def generate_persona_names(summary_df):
    """Sends all cluster data to Gemini 2.5 Flash to get persona names/descriptions."""
    model = genai.GenerativeModel('gemini-2.0-flash') # Using latest available flash
    
    # Create a string representation of the clusters for the prompt
    cluster_info = summary_df.to_string()
    
    prompt = f"""
    Based on the following US Census cluster data, create a creative Persona Name (e.g., 'The Struggling Graduate' or 'Wealthy Empty-Nesters') 
    and a short 1-sentence description for each Cluster ID.
    
    Data:
    {cluster_info}
    
    Return the result ONLY as a JSON list of objects with keys: "Cluster", "Persona Name", "Description".
    Ensure "Cluster" matches the IDs provided.
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean the response text in case Gemini adds markdown backticks
        json_text = response.text.replace('```json', '').replace('```', '').strip()
        persona_data = json.loads(json_text)
        return pd.DataFrame(persona_data)
    except Exception as e:
        st.error(f"GenAI Error: {e}")
        return pd.DataFrame()

# --- 3. Main Streamlit App ---

st.title("US Population Personas Clustering")

st.sidebar.header("Configuration")
if "CENSUS_API_KEY" in st.secrets:
    api_key = st.secrets["CENSUS_API_KEY"]
else:
    st.sidebar.error("CENSUS_API_KEY not found in secrets.")
    st.stop()

selected_state = st.sidebar.selectbox("Select US State", list(STATE_FIPS.keys()))
n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 8, 3)

if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'summary' not in st.session_state:
    st.session_state['summary'] = None

if st.sidebar.button("Execute"):
    with st.spinner(f"Fetching data, creating clusters and generating personas..."):
        fips = STATE_FIPS[selected_state]
        raw_df = fetch_pums_data(fips, api_key)
        
        if not raw_df.empty:
            df = process_data(raw_df)
            features = ['AGEP', 'PINCP', 'HHT', 'SCHL']
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[features])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(scaled_features)
            
            # --- Generate Summary and GenAI Personas ---
            summary = df.groupby('Cluster').agg({
                'PINCP': 'mean',
                'AGEP': 'mean',
                'Education Level': lambda x: x.mode()[0] if not x.mode().empty else "N/A",
                'Household Type': lambda x: x.mode()[0] if not x.mode().empty else "N/A"
            }).reset_index()

            # Get AI Personas
            ai_personas = generate_persona_names(summary)
            
            if not ai_personas.empty:
                # Merge AI results into summary
                ai_personas['Cluster'] = ai_personas['Cluster'].astype(int)
                summary = summary.merge(ai_personas, on='Cluster')
                
                # Map the Persona Names back to the main dataframe for the Legend
                persona_map = dict(zip(summary['Cluster'], summary['Persona Name']))
                df['Persona Name'] = df['Cluster'].map(persona_map)
            else:
                df['Persona Name'] = df['Cluster'].astype(str)

            st.session_state['data'] = df
            st.session_state['summary'] = summary
            st.success(f"Loaded {len(df)} records. AI Personas generated!")

# --- Display Logic ---
if st.session_state['data'] is not None:
    df = st.session_state['data']
    summary = st.session_state['summary']
    
    st.divider()
    st.subheader("📊 Cluster Visualizations")
    
    plot_vars = {'Age': 'AGEP', 'Income': 'PINCP', 'Education': 'Education Level', 'Household Type': 'Household Type'}
    combinations = list(itertools.combinations(plot_vars.keys(), 2))
    combo_labels = [f"{x} vs {y}" for x, y in combinations]
    selected_combo = st.selectbox("Select Graph to Show:", combo_labels)
    x_label, y_label = next((x, y) for x, y in combinations if f"{x} vs {y}" == selected_combo)
    
    plot_df = df.sample(min(2000, len(df))).copy()
    
    # Use 'Persona Name' for the color (Legend)
    fig = px.scatter(
        plot_df, 
        x=plot_vars[x_label], 
        y=plot_vars[y_label],
        color='Persona Name', # This ensures names show in the legend
        title=f"Persona Clusters: {x_label} vs {y_label}",
        hover_data=['Age Range', 'Education Level', 'Household Type'],
        labels={
            'AGEP': 'Age (Years)',
            'PINCP': 'Annual Income ($)',
            'Education Level': 'Education Degree',
            'Persona Name': 'Detected Persona'
        },
        category_orders={'Education Level': EDU_ORDER}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("📋 Persona Summary")
    
    # Format summary for display
    display_summary = summary.copy()
    display_summary['PINCP'] = display_summary['PINCP'].round().astype(int).apply(lambda x: f"${x:,}")
    display_summary['AGEP'] = display_summary['AGEP'].round().astype(int)
    
    st.table(display_summary)
