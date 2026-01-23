import streamlit as st
import pandas as pd
import us
from census import Census
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Census Clusters", layout="wide")

# --- LOAD SECRETS ---
# Modified to look for top-level CENSUS_API and GOOGLE_API_KEY
try:
    CENSUS_API_KEY = st.secrets["CENSUS_API"]
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("Secrets file not found. Please check your .streamlit/secrets.toml or Streamlit Cloud settings.")
    st.stop()
except KeyError as e:
    st.error(f"Missing API key in secrets: {e}. Please ensure 'CENSUS_API' and 'GOOGLE_API_KEY' are set.")
    st.stop()

# --- CONFIGURE AI ---
genai.configure(api_key=GEMINI_API_KEY)

# --- CACHED DATA FUNCTIONS ---
@st.cache_data
def get_census_data(state_fips, api_key):
    """
    Fetches ZCTA (Zip Code) data for a specific state.
    """
    c = Census(api_key)
    
    # Variables: Total Pop, Median Income, Median Age
    variables = ['B01003_001E', 'B19013_001E', 'B01002_001E', 'NAME']
    
    # Query: Get all Zip Code Tabulation Areas (ZCTA) for the specific state
    geo_filter = {'for': 'zip code tabulation area:*', 'in': f'state:{state_fips}'}
    
    try:
        data = c.acs5.get(variables, geo=geo_filter)
        df = pd.DataFrame(data)
        
        # Rename columns
        rename_map = {
            'B01003_001E': 'Population',
            'B19013_001E': 'Median_Income',
            'B01002_001E': 'Median_Age',
            'zip code tabulation area': 'Zip_Code'
        }
        df = df.rename(columns=rename_map)
        
        # Convert numeric columns
        numeric_cols = ['Population', 'Median_Income', 'Median_Age']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Error fetching Census data: {e}")
        return pd.DataFrame()

def generate_persona(cluster_stats):
    """
    Uses Gemini to generate a persona based on cluster statistics.
    """
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    You are a demographic expert. I have identified a population cluster with the following average statistics:
    - Average Median Income: ${cluster_stats['Median_Income']:,.0f}
    - Average Median Age: {cluster_stats['Median_Age']:.1f} years old
    - Average Population per Zip: {cluster_stats['Population']:,.0f}
    
    Task:
    1. Create a creative "Persona Name" for this group.
    2. Write a one-sentence description of their lifestyle.
    
    Format the output exactly like this:
    Name: [Name]
    Description: [Description]
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Name: Unknown\nDescription: AI Error - {e}"

# --- SIDEBAR UI ---
st.sidebar.header("Configuration")

# State Selector
state_list = [s.name for s in us.states.STATES]
selected_state_name = st.sidebar.selectbox("Select State", state_list, index=state_list.index("Illinois"))
selected_state_obj = us.states.lookup(selected_state_name)

# Cluster Selector
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=6, value=3)

# Population Filter
min_pop = st.sidebar.number_input("Minimum Population per Zip", min_value=0, value=1000, step=500)

# --- MAIN APP ---
st.title(f"📍 Demographic Clustering: {selected_state_name}")
st.markdown("Fetching Census data, clustering Zip Codes, and generating AI personas.")

# 1. Fetch Data
with st.spinner('Querying US Census Bureau API...'):
    raw_df = get_census_data(selected_state_obj.fips, CENSUS_API_KEY)

if raw_df.empty:
    st.warning("No data found or API error. Check your API key.")
    st.stop()

# 2. Preprocessing
filtered_df = raw_df[raw_df['Population'] >= min_pop].dropna()
st.write(f"Analyzing **{len(filtered_df)}** Zip Codes in {selected_state_name} (after filtering).")

if len(filtered_df) < n_clusters:
    st.error("Not enough data points for the requested number of clusters.")
    st.stop()

# 3. Clustering (K-Means)
features = ['Median_Income', 'Median_Age', 'Population']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(filtered_df[features])

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
filtered_df['Cluster'] = kmeans.fit_predict(scaled_data)

# 4. Generate Personas & Stats
st.divider()
st.subheader("🧩 AI-Generated Cluster Personas")

cluster_summary = filtered_df.groupby('Cluster')[features].mean().reset_index()
cols = st.columns(n_clusters)

for index, row in cluster_summary.iterrows():
    with cols[index]:
        with st.spinner(f'Generating persona {index+1}...'):
            ai_response = generate_persona(row)
            try:
                name = ai_response.split("Name:")[1].split("Description:")[0].strip()
                desc = ai_response.split("Description:")[1].strip()
            except:
                name = f"Cluster {index}"
                desc = ai_response

            st.success(f"**{name}**")
            st.caption(desc)
            st.markdown(f"""
            * **Avg Income:** ${row['Median_Income']:,.0f}
            * **Avg Age:** {row['Median_Age']:.1f}
            * **Avg Pop:** {row['Population']:,.0f}
            """)

# 5. Visualizations
st.divider()
st.subheader("📊 Cluster Visualization")

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=filtered_df, 
        x='Median_Age', 
        y='Median_Income', 
        hue='Cluster', 
        palette='viridis', 
        s=100, 
        alpha=0.7,
        ax=ax
    )
    plt.title("Zip Codes: Income vs. Age")
    plt.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

with col2:
    st.markdown("**Top Zips in Cluster 0**")
    st.dataframe(
        filtered_df[filtered_df['Cluster'] == 0]
        .sort_values('Median_Income', ascending=False)
        .head(10)[['Zip_Code', 'Median_Income', 'Median_Age']]
        .reset_index(drop=True),
        use_container_width=True
    )
