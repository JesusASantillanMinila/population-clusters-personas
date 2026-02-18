US Population Persona Segmenter
This repository contains a professional-grade analytical tool designed to transform raw demographic data into actionable market segments. By integrating Unsupervised Machine Learning with Large Language Models (LLMs), the application automates the end-to-end process of data retrieval, statistical clustering, and persona generation.

Project Overview
The application identifies distinct socio-economic clusters within US state populations. It leverages the U.S. Census Bureau’s Public Use Microdata Sample (PUMS) to ensure high-fidelity modeling based on verified federal data.

Core Methodology
1. Statistical Clustering (K-Means)
The engine utilizes a K-Means Clustering algorithm to segment the population. By analyzing multi-dimensional features—including annual income, age, education level, and household structure—the model identifies natural groupings within the dataset. To ensure statistical integrity, the data is preprocessed using StandardScaler to normalize features and prevent high-magnitude variables from biasing the centroids.

2. Generative AI Synthesis
Once clusters are established, the statistical summaries (means and modes) are passed to Google Gemini 2.5 Flash. The LLM acts as an interpretive layer, translating abstract cluster data into human-readable personas. This eliminates manual bias and provides immediate context to the quantitative results.

Technical Specifications
Interface: Streamlit

Machine Learning: Scikit-Learn (KMeans, StandardScaler)

Generative AI: Google Generative AI API

Data Visualization: Plotly Express (Scatter and Mapbox)

Data Source: 2022 American Community Survey (ACS) 1-Year PUMS API

Prerequisites
Python 3.9 or higher

U.S. Census API Key

Google Gemini API Key
