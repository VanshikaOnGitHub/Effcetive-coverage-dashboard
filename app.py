#!/usr/bin/env python
# coding: utf-8

# In[29]:


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Effective Coverage Dashboard", initial_sidebar_state="expanded")

# --- Function to load data (handles file upload) ---
@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the CSV data."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=False)
            df['Year'] = df['Year'].astype(str)
            df['Month'] = df['Month'].astype(str)
            
            if 'ECO' in df.columns:
                df['ECO'] = pd.to_numeric(df['ECO'], errors='coerce')
            
            df.dropna(subset=['Year', 'Month', 'ECO', 'Zone', 'Town', 'State', 'Revised Channel', 'ASM'], inplace=True)
            return df
        except Exception as e:
            st.error(f"Error loading or processing file: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# --- Streamlit Application Layout ---
st.title("📊 Effective Coverage (ECO) Real-Time Dashboard")

# 1. File Uploader and Data Loading
uploaded_file = st.file_uploader("Upload your Effective Coverage CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
else:
    df = load_data(uploaded_file)

    if df.empty:
        st.stop()

    # --- Sidebar for Global Filters (Slicers) ---
    st.sidebar.header("Global Filters")

    # Get unique values for filters
    all_years = sorted(df['Year'].unique())
    all_months = sorted(df['Month'].unique())
    all_rev_channels = sorted(df['Revised Channel'].unique())

    # Add "All" option to Year filter
    year_options = ["All"] + all_years
    selected_years_input = st.sidebar.multiselect("Select Year(s)", year_options, default=["All"])
    
    # Handle "All" selection for years
    if "All" in selected_years_input:
        selected_years = all_years
    else:
        selected_years = selected_years_input

    # Add "All" option to Month filter
    month_options = ["All"] + all_months
    selected_months_input = st.sidebar.multiselect("Select Month(s)", month_options, default=["All"])
    
    # Handle "All" selection for months
    if "All" in selected_months_input:
        selected_months = all_months
    else:
        selected_months = selected_months_input

    # Revised Channel filter
    selected_channels = st.sidebar.multiselect("Select Revised Channel(s)", all_rev_channels, default=all_rev_channels)

    # Apply global filters
    df_filtered = df[
        df['Year'].isin(selected_years) &
        df['Month'].isin(selected_months) &
        df['Revised Channel'].isin(selected_channels)
    ]

    if df_filtered.empty:
        st.warning("No data matches the selected global filters.")
        st.stop()

    # --- 1. Zone-wise Coverage Pie Chart ---
    st.header("1. Zone-wise Coverage Distribution")
    
    zone_data = df_filtered.groupby('Zone')['ECO'].sum().reset_index()

    fig1 = px.pie(
        zone_data,
        values='ECO',
        names='Zone',
        title='Total Effective Coverage by Zone',
        hole=.3,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # --- 2. ECO Trends Line Graph ---
    st.header("2. Effective Coverage (ECO) Trends Over Months")

    # Dynamic filters for the Line Graph with cascading logic
    colA, colB, colC, colD = st.columns(4)
    
    with colA:
        all_asms = sorted(df_filtered['ASM'].unique())
        asm_options = ["All"] + all_asms
        selected_asms_input = st.multiselect("Filter by ASM", asm_options, default=["All"], key='asm_filter')
        
        # Handle "All" selection
        if "All" in selected_asms_input:
            selected_asms = all_asms
        else:
            selected_asms = selected_asms_input

    # Filter data based on ASM selection for cascading filters
    df_for_state = df_filtered[df_filtered['ASM'].isin(selected_asms)]
    
    with colB:
        all_states = sorted(df_for_state['State'].unique())
        state_options = ["All"] + all_states
        selected_states_input = st.multiselect("Filter by State", state_options, default=["All"], key='state_filter')
        
        # Handle "All" selection
        if "All" in selected_states_input:
            selected_states = all_states
        else:
            selected_states = selected_states_input

    # Filter data based on State selection for town filter
    df_for_town = df_for_state[df_for_state['State'].isin(selected_states)]
    
    with colC:
        all_towns = sorted(df_for_town['Town'].unique())
        town_options = ["All"] + all_towns
        selected_towns_input = st.multiselect("Filter by Town", town_options, default=["All"], key='town_filter')
        
        # Handle "All" selection
        if "All" in selected_towns_input:
            selected_towns = all_towns
        else:
            selected_towns = selected_towns_input

    with colD:
        all_trend_channels = sorted(df_filtered['Revised Channel'].unique())
        channel_options = ["All"] + all_trend_channels
        selected_trend_channels_input = st.multiselect("Filter by Revised Channel", channel_options, default=["All"], key='channel_filter')
        
        # Handle "All" selection
        if "All" in selected_trend_channels_input:
            selected_trend_channels = all_trend_channels
        else:
            selected_trend_channels = selected_trend_channels_input

    # Apply line graph filters
    df_trend = df_filtered[
        df_filtered['ASM'].isin(selected_asms) &
        df_filtered['State'].isin(selected_states) &
        df_filtered['Town'].isin(selected_towns) &
        df_filtered['Revised Channel'].isin(selected_trend_channels)
    ]

    if df_trend.empty:
        st.warning("No data matches the selected filters for the ECO Trend chart.")
    else:
        trend_data = df_trend.groupby(['Month'])['ECO'].sum().reset_index()
        
        fig2 = px.line(
            trend_data,
            x='Month',
            y='ECO',
            title='ECO Trend Over Months',
            markers=True
        )
        fig2.update_layout(xaxis_title="Month", yaxis_title="Total ECO")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # --- 3. Top 5 / Bottom 5 Towns Bar Graph with Gradient ---
    st.header("3. Top 5 and Bottom 5 Towns by ECO")

    df_bar_chart = df_filtered.copy()
    
    grouped_towns = df_bar_chart.groupby(['Town', 'State']).agg(
        Total_ECO=('ECO', 'sum'),
        ASMs=('ASM', lambda x: ', '.join(sorted(x.unique())))
    ).reset_index()
    
    top_5 = grouped_towns.sort_values(by='Total_ECO', ascending=False).head(5).copy()
    bottom_5 = grouped_towns.sort_values(by='Total_ECO', ascending=True).head(5).copy()
    
    towns_data = pd.concat([top_5, bottom_5]).drop_duplicates(subset=['Town', 'State'])
    
    # Sort by Total_ECO descending for proper gradient application
    towns_data = towns_data.sort_values(by='Total_ECO', ascending=False).reset_index(drop=True)
    
    # Create gradient colors from green to red (light to dark)
    # Top 5 will be shades of green, Bottom 5 will be shades of red
    n = len(towns_data)
    colors = []
    
    for i, row in towns_data.iterrows():
        if row['Town'] in top_5['Town'].values:
            # Green gradient: darker green for #1, lighter for #5
            position = list(top_5.sort_values(by='Total_ECO', ascending=False)['Town']).index(row['Town'])
            # RGB for green gradient: from dark green (0,100,0) to light green (144,238,144)
            intensity = 100 + (position * 27)  # 100, 127, 154, 181, 208
            colors.append(f'rgb(0,{intensity},0)')
        else:
            # Red gradient: darker red for lowest, lighter for higher
            position = list(bottom_5.sort_values(by='Total_ECO', ascending=True)['Town']).index(row['Town'])
            # RGB for red gradient: from dark red (139,0,0) to light red (255,182,193)
            r_val = 139 + (position * 23)  # 139, 162, 185, 208, 231
            g_val = position * 36  # 0, 36, 72, 108, 144
            b_val = position * 38  # 0, 38, 76, 114, 152
            colors.append(f'rgb({r_val},{g_val},{b_val})')
    
    # Create bar chart with custom colors
    fig3 = go.Figure(data=[
        go.Bar(
            x=towns_data['Town'],
            y=towns_data['Total_ECO'],
            marker_color=colors,
            text=towns_data['Total_ECO'].round(0),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>State: %{customdata[0]}<br>Total ECO: %{y:.0f}<br>ASMs: %{customdata[1]}<extra></extra>',
            customdata=towns_data[['State', 'ASMs']].values
        )
    ])
    
    fig3.update_layout(
        title='Top 5 and Bottom 5 Towns by Total ECO',
        xaxis_title="Town",
        yaxis_title="Total ECO",
        showlegend=False,
        height=500
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Display the underlying ASM information
    st.subheader("ASM Information for Top/Bottom Towns")
    
    # Add ranking
    display_data = towns_data.copy()
    display_data['Rank'] = range(1, len(display_data) + 1)
    display_data['Category'] = display_data['Town'].apply(
        lambda x: '🟢 Top 5' if x in top_5['Town'].values else '🔴 Bottom 5'
    )
    
    st.dataframe(
        display_data[['Rank', 'Category', 'Town', 'State', 'Total_ECO', 'ASMs']].rename(
            columns={'Total_ECO': 'Total ECO', 'ASMs': 'Associated ASM(s)'}
        ),
        use_container_width=True,
        hide_index=True
    )


# In[31]:


get_ipython().system('jupyter nbconvert --to script app.ipynb')


# In[ ]:




