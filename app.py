#!/usr/bin/env python
# coding: utf-8

# In[29]:


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Effective Coverage Dashboard", initial_sidebar_state="expanded")

# --- Function to load data from local CSV file ---
@st.cache_data
def load_data():
    """Loads and preprocesses the CSV data from local file."""
    try:
        # Update this path to your CSV file location
        df = pd.read_csv(r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\GTM TRENDS\ECO Trends.csv", encoding='utf-8')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=False)
        
        # Convert to appropriate types
        df['Year'] = df['Year'].astype(str)
        df['Month'] = df['Month'].astype(str)
        
        # Ensure ECO is numeric
        if 'ECO' in df.columns:
            df['ECO'] = pd.to_numeric(df['ECO'], errors='coerce')
        
        # Drop rows with missing essential data
        required_cols = ['Year', 'Month', 'ECO', 'Zone', 'Town', 'State', 'Revised Channel', 'ASM']
        if 'SKU Type' in df.columns:
            required_cols.append('SKU Type')
        
        df.dropna(subset=[col for col in required_cols if col in df.columns], inplace=True)
        
        return df
    except FileNotFoundError:
        st.error("Error: 'effective_coverage_data.csv' file not found. Please ensure the file is in the same directory as this script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
        return pd.DataFrame()

# --- Streamlit Application Layout ---
st.title("📊 Effective Coverage (ECO) Real-Time Dashboard")

# Load data automatically
with st.spinner('Loading data...'):
    df = load_data()

if df.empty:
    st.error("Unable to load data. Please check the CSV file.")
    st.stop()

st.success(f"✅ Data loaded successfully! Total records: {len(df):,}")

# --- Sidebar for Global Filters (Slicers) ---
st.sidebar.header("Global Filters")

# Get unique values for filters
all_years = sorted(df['Year'].unique())
all_months = sorted(df['Month'].unique())
all_rev_channels = sorted(df['Revised Channel'].unique())

# Year filter with "All" option
year_options = ["All"] + all_years
selected_years_input = st.sidebar.multiselect("Select Year(s)", year_options, default=["All"])
selected_years = all_years if "All" in selected_years_input else selected_years_input

# Month filter with "All" option
month_options = ["All"] + all_months
selected_months_input = st.sidebar.multiselect("Select Month(s)", month_options, default=["All"])
selected_months = all_months if "All" in selected_months_input else selected_months_input

# Revised Channel filter with "All" option
channel_options = ["All"] + all_rev_channels
selected_channels_input = st.sidebar.multiselect("Select Revised Channel(s)", channel_options, default=["All"])
selected_channels = all_rev_channels if "All" in selected_channels_input else selected_channels_input

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

# Dynamic filters with cascading logic
colA, colB, colC, colD = st.columns(4)

with colA:
    all_asms = sorted(df_filtered['ASM'].unique())
    asm_options = ["All"] + all_asms
    selected_asms_input = st.multiselect("Filter by ASM", asm_options, default=["All"], key='asm_filter')
    selected_asms = all_asms if "All" in selected_asms_input else selected_asms_input

# Cascading filter
df_for_state = df_filtered[df_filtered['ASM'].isin(selected_asms)]

with colB:
    all_states = sorted(df_for_state['State'].unique())
    state_options = ["All"] + all_states
    selected_states_input = st.multiselect("Filter by State", state_options, default=["All"], key='state_filter')
    selected_states = all_states if "All" in selected_states_input else selected_states_input

df_for_town = df_for_state[df_for_state['State'].isin(selected_states)]

with colC:
    all_towns = sorted(df_for_town['Town'].unique())
    town_options = ["All"] + all_towns
    selected_towns_input = st.multiselect("Filter by Town", town_options, default=["All"], key='town_filter')
    selected_towns = all_towns if "All" in selected_towns_input else selected_towns_input

with colD:
    all_trend_channels = sorted(df_filtered['Revised Channel'].unique())
    channel_options2 = ["All"] + all_trend_channels
    selected_trend_channels_input = st.multiselect("Filter by Revised Channel", channel_options2, default=["All"], key='channel_filter')
    selected_trend_channels = all_trend_channels if "All" in selected_trend_channels_input else selected_trend_channels_input

# Apply filters
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
towns_data = towns_data.sort_values(by='Total_ECO', ascending=False).reset_index(drop=True)

# Create gradient colors
colors = []
for i, row in towns_data.iterrows():
    if row['Town'] in top_5['Town'].values:
        position = list(top_5.sort_values(by='Total_ECO', ascending=False)['Town']).index(row['Town'])
        intensity = 100 + (position * 27)
        colors.append(f'rgb(0,{intensity},0)')
    else:
        position = list(bottom_5.sort_values(by='Total_ECO', ascending=True)['Town']).index(row['Town'])
        r_val = 139 + (position * 23)
        g_val = position * 36
        b_val = position * 38
        colors.append(f'rgb({r_val},{g_val},{b_val})')

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

# Display ASM information
st.subheader("ASM Information for Top/Bottom Towns")

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

st.markdown("---")

# --- 4. ECO by Revised Channel with SKU Type Filter ---
st.header("4. ECO Distribution by Revised Channel")

# Check if SKU Type column exists
has_sku_type = 'SKU Type' in df_filtered.columns

# Filters for channel visualization
col1, col2, col3, col4 = st.columns(4)

if has_sku_type:
    with col1:
        all_sku_types = sorted(df_filtered['SKU Type'].unique())
        sku_options = ["All"] + all_sku_types
        selected_sku_input = st.multiselect("Filter by SKU Type", sku_options, default=["All"], key='sku_filter')
        selected_sku = all_sku_types if "All" in selected_sku_input else selected_sku_input
    
    df_for_channel = df_filtered[df_filtered['SKU Type'].isin(selected_sku)]
else:
    df_for_channel = df_filtered.copy()

with col2:
    all_asms_ch = sorted(df_for_channel['ASM'].unique())
    asm_options_ch = ["All"] + all_asms_ch
    selected_asms_ch_input = st.multiselect("Filter by ASM", asm_options_ch, default=["All"], key='asm_channel_filter')
    selected_asms_ch = all_asms_ch if "All" in selected_asms_ch_input else selected_asms_ch_input

df_for_state_ch = df_for_channel[df_for_channel['ASM'].isin(selected_asms_ch)]

with col3:
    all_states_ch = sorted(df_for_state_ch['State'].unique())
    state_options_ch = ["All"] + all_states_ch
    selected_states_ch_input = st.multiselect("Filter by State", state_options_ch, default=["All"], key='state_channel_filter')
    selected_states_ch = all_states_ch if "All" in selected_states_ch_input else selected_states_ch_input

df_for_town_ch = df_for_state_ch[df_for_state_ch['State'].isin(selected_states_ch)]

with col4:
    all_towns_ch = sorted(df_for_town_ch['Town'].unique())
    town_options_ch = ["All"] + all_towns_ch
    selected_towns_ch_input = st.multiselect("Filter by Town", town_options_ch, default=["All"], key='town_channel_filter')
    selected_towns_ch = all_towns_ch if "All" in selected_towns_ch_input else selected_towns_ch_input

# Apply all filters
df_channel = df_for_channel[
    df_for_channel['ASM'].isin(selected_asms_ch) &
    df_for_channel['State'].isin(selected_states_ch) &
    df_for_channel['Town'].isin(selected_towns_ch)
]

if df_channel.empty:
    st.warning("No data matches the selected filters for Channel distribution.")
else:
    # Aggregate by Revised Channel
    channel_eco_data = df_channel.groupby('Revised Channel')['ECO'].sum().reset_index()
    channel_eco_data = channel_eco_data.sort_values(by='ECO', ascending=False)
    
    # Create bar chart
    fig4 = px.bar(
        channel_eco_data,
        x='Revised Channel',
        y='ECO',
        title='Total ECO by Revised Channel',
        color='ECO',
        color_continuous_scale='Viridis',
        text='ECO'
    )
    
    fig4.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig4.update_layout(
        xaxis_title="Revised Channel",
        yaxis_title="Total ECO",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Summary table
    st.subheader("Channel-wise ECO Summary")
    channel_eco_data['Percentage'] = (channel_eco_data['ECO'] / channel_eco_data['ECO'].sum() * 100).round(2)
    
    st.dataframe(
        channel_eco_data.rename(columns={'ECO': 'Total ECO', 'Percentage': '% Share'}),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")
st.caption("Dashboard created with Streamlit | Data updates automatically from CSV file")







