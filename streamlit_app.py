import streamlit as st
import pandas as pd
from census import Census
import us
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import google.generativeai as genai

# --- Configuration & Setup ---
st.set_page_config(page_title="Census Persona Clustering", layout="wide")

st.title("🏘️ Census Demographic Clustering & AI Personas")
st.markdown("""
This app retrieves real-time US Census data to cluster counties based on **Age** and **Household Income**.
It then uses **Gemini 2.5** to generate unique personas for each demographic cluster.
""")

# --- Helper: State to FIPS Mapping ---
# Creates a dictionary like {'Alabama': '01', ...}
states_mapping = {state.name: state.fips for state in us.states.STATES}

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

# 1. State Selection
selected_state_name = st.sidebar.selectbox("Select a State", list(states_mapping.keys()))
selected_state_fips = states_mapping[selected_state_name]

# 2. Clustering Parameters
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=8, value=3)
min_pop = st.sidebar.number_input("Minimum County Population", min_value=0, value=10000, step=1000)

# --- Check Secrets ---
if "CENSUS_API_KEY" not in st.secrets or "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing API Keys in `st.secrets`. Please add CENSUS_API_KEY and GOOGLE_API_KEY.")
    st.stop()

# --- Data Loading Function ---
@st.cache_data
def get_census_data(api_key, state_fips):
    c = Census(api_key)
    
    # Variables: 
    # B01003_001E: Total Population
    # B19013_001E: Median Household Income
    # B01002_001E: Median Age
    variables = ('NAME', 'B01003_001E', 'B19013_001E', 'B01002_001E')
    
    # Fetch data for all counties in the selected state
    # acs5 represents the 5-year American Community Survey
    data = c.acs5.state_county(fields=variables, state_fips=state_fips, county_fips="*")
    
    df = pd.DataFrame(data)
    
    # Rename columns for clarity
    df = df.rename(columns={
        'NAME': 'County Name',
        'B01003_001E': 'Population',
        'B19013_001E': 'Median Income',
        'B01002_001E': 'Median Age'
    })
    
    # Convert numeric columns to appropriate types and drop rows with missing data
    cols = ['Population', 'Median Income', 'Median Age']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=cols)
    return df

# --- Main Execution ---

# 1. Load Data
try:
    with st.spinner(f"Fetching Census data for {selected_state_name}..."):
        df_raw = get_census_data(st.secrets["CENSUS_API_KEY"], selected_state_fips)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# 2. Filter by Population
df_filtered = df_raw[df_raw['Population'] >= min_pop].copy()

if df_filtered.empty:
    st.warning("No counties match the population criteria. Please lower the threshold.")
    st.stop()

st.write(f"Analyzing **{len(df_filtered)}** counties in {selected_state_name} with population > {min_pop:,}")

# 3. Clustering Logic
# We need to scale the data because Income ($50k+) and Age (30-50) have vastly different ranges
scaler = StandardScaler()
features = ['Median Income', 'Median Age']
X = df_filtered[features]
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. Prepare Summary for AI
# Calculate the mean Income and Age for each cluster to send to Gemini
cluster_summary = df_filtered.groupby('Cluster')[features].mean().reset_index()
cluster_counts = df_filtered['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']
cluster_summary = cluster_summary.merge(cluster_counts, on='Cluster')

# --- GenAI Persona Generation ---

def generate_personas(summary_df, state):
    # Setup Gemini
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    
    # Using the specific template requested
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="You are a demographic expert. Your goal is to create creative, distinct, and descriptive personas for population clusters based on their median age and income.",
        tools=[{"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "unspecified"}}}]  # Fallback
    )
    
    # Construct the prompt
    prompt = f"""
    I have clustered counties in {state} into {len(summary_df)} groups based on Median Household Income and Median Age.
    
    Here is the data for each cluster:
    {summary_df.to_string(index=False)}
    
    For each cluster (0 to {len(summary_df)-1}), provide:
    1. A catchy Persona Name (e.g., "Retiring Wealthy", "Young Professionals").
    2. A brief 1-sentence description.
    3. Key attributes (High/Low Income, Young/Old).
    
    Return the answer as a valid JSON object where keys are the Cluster numbers (as strings) and values contain 'name', 'description', and 'attributes'.
    Do not use markdown formatting like ```json. Just return the raw JSON string.
    """
    
    with st.spinner("Asking Gemini to generate personas..."):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"GenAI Error: {e}")
            return None

# Generate Personas
ai_response_text = generate_personas(cluster_summary, selected_state_name)

# Parse JSON response (Basic cleaning to ensure valid JSON)
import json
import re

personas = {}
if ai_response_text:
    try:
        # Strip potential markdown code blocks
        clean_json = re.sub(r'```json|```', '', ai_response_text).strip()
        personas = json.loads(clean_json)
    except json.JSONDecodeError:
        st.warning("Could not parse AI response as JSON. Displaying raw output below.")
        st.write(ai_response_text)

# --- visualization ---

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Cluster Visualization")
    
    # Map cluster numbers to AI names if available for the legend
    df_filtered['Cluster Name'] = df_filtered['Cluster'].apply(
        lambda x: personas.get(str(x), {}).get('name', f'Cluster {x}') if personas else f'Cluster {x}'
    )
    
    fig = px.scatter(
        df_filtered,
        x="Median Age",
        y="Median Income",
        color="Cluster Name",
        hover_data=["County Name", "Population"],
        title=f"Demographic Clusters in {selected_state_name}",
        size="Population", 
        size_max=40
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("AI Generated Personas")
    if personas:
        for cluster_id, details in personas.items():
            st.markdown(f"### {details.get('name')}")
            st.markdown(f"**Cluster {cluster_id}**")
            st.info(details.get('description'))
            st.caption(f"**Attributes:** {details.get('attributes')}")
            
            # Show stats for this specific cluster from our calculated summary
            stats = cluster_summary[cluster_summary['Cluster'] == int(cluster_id)].iloc[0]
            st.markdown(f"""
            * **Avg Income:** ${stats['Median Income']:,.0f}
            * **Avg Age:** {stats['Median Age']:.1f}
            * **Counties:** {int(stats['Count'])}
            """)
            st.divider()
    else:
        st.write("No persona data generated.")

# --- Data Table ---
with st.expander("View Raw Data"):
    st.dataframe(df_filtered)
