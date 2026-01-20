import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os

# --- Page Config ---
st.set_page_config(
    page_title="Financial Data Visualizer",
    page_icon="fq",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styles (Dark Mode friendly) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading Function ---
@st.cache_data(show_spinner=True)
def load_data(data_path="./csv_raw/*.csv"):
    all_files = glob.glob(data_path)
    
    if not all_files:
        return None

    df_list = []
    for filename in all_files:
        try:
            # Optimize data types for memory efficiency if possible, 
            # but reading as objects/floats first is safer.
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            continue

    if not df_list:
        return None

    # Filter out empty or None DataFrames
    df_list = [d for d in df_list if d is not None and not d.empty]

    if not df_list:
        return None

    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    
    # Convert date columns to datetime
    date_cols = ['datadate', 'rdq']
    for col in date_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce')
    
    # Clean numeric columns
    numeric_keywords = ['price', 'yield', 'volume', 'return']
    for col in combined_df.columns:
        if any(keyword in col for keyword in numeric_keywords):
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    # Fill NaN values in volume columns with 0 to prevent Plotly errors
    volume_cols = [c for c in combined_df.columns if 'volume' in c]
    if volume_cols:
        combined_df[volume_cols] = combined_df[volume_cols].fillna(0)
            
    return combined_df

# --- Main App ---
def main():
    st.title("📊 Financial Data Visualizer")
    
    with st.spinner('Loading heavy datasets... This might take a moment.'):
        df = load_data()

    if df is None:
        st.warning("No CSV files found in `./csv_raw`. Please add files to analyze.")
        return

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filters Setup")
    
    # Fiscal Year Filter
    if 'fyearq' in df.columns:
        years = sorted(df['fyearq'].unique())
        selected_years = st.sidebar.multiselect("Select Fiscal Year(s)", years, default=years)
        if selected_years:
            df = df[df['fyearq'].isin(selected_years)]

    # Ticker Filter (TIC)
    if 'tic' in df.columns:
        tickers = sorted(df['tic'].astype(str).unique())
        
        # Initialize session state for tickers if not present
        if 'selected_tickers_state' not in st.session_state:
            st.session_state['selected_tickers_state'] = tickers[:3] if len(tickers) > 0 else []

        st.sidebar.subheader("Ticker Selection")
        col_t1, col_t2 = st.sidebar.columns(2)
        
        if col_t1.button("Select All"):
            st.session_state['selected_tickers_state'] = tickers
        
        if col_t2.button("Deselect All"):
            st.session_state['selected_tickers_state'] = []
            
        selected_tickers = st.sidebar.multiselect(
            "Select Ticker(s)", 
            tickers, 
            key='selected_tickers_state'
        )
        
        if selected_tickers:
            df = df[df['tic'].isin(selected_tickers)]
    
    # Date Range Filter (based on rdq)
    if 'rdq' in df.columns:
        min_date = df['rdq'].min().date()
        max_date = df['rdq'].max().date()
        
        # Ensure dates are valid
        if pd.notnull(min_date) and pd.notnull(max_date):
            start_date, end_date = st.sidebar.date_input(
                "Select Report Date Range", 
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
            df = df[(df['rdq'].dt.date >= start_date) & (df['rdq'].dt.date <= end_date)]

    st.sidebar.markdown("---")
    st.sidebar.info(f"Showing **{len(df)}** rows")

    # --- Dashboard Content ---
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Unique Tickers", df['tic'].nunique() if 'tic' in df.columns else 0)
    col3.metric("Avg Bond Return", f"{df['bond_return'].mean():.4f}" if 'bond_return' in df.columns else "N/A")
    col4.metric("Unique Sectors", df['gsector'].nunique() if 'gsector' in df.columns else 0)

    # Tabs for Organization
    tab1, tab2, tab3 = st.tabs(["📈 Price & Yield", "📊 Distributions", "📋 Raw Data"])

    with tab1:
        st.subheader("Price & Yield Analysis")
        
        if 'tic' in df.columns and len(selected_tickers) > 0:
             # Price vs Time (rdq_price)
            if 'rdq_price' in df.columns and 'rdq' in df.columns:
                fig_price = px.line(
                    df.sort_values(by='rdq'), 
                    x='rdq', 
                    y='rdq_price', 
                    color='tic',
                    markers=True,
                    title="Price Trends over Time (RDQ Price)",
                    labels={'rdq': 'Report Date', 'rdq_price': 'Price'}
                )
                st.plotly_chart(fig_price, use_container_width=True)
            
            # Yield comparison 
            if 'rdq_yield' in df.columns and 'rdq' in df.columns:
                 fig_yield = px.scatter(
                    df, 
                    x='rdq', 
                    y='rdq_yield', 
                    size='rdq_volume' if 'rdq_volume' in df.columns else None,
                    color='tic',
                    title="Yield Analysis (sized by Volume)",
                    hover_data=['tic', 'rdq', 'rdq_price']
                )
                 st.plotly_chart(fig_yield, use_container_width=True)
        else:
            st.info("Select specific tickers to view trend lines clearly.")

    with tab2:
        st.subheader("Distribution Analysis")
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            if 'bond_return' in df.columns:
                fig_hist = px.histogram(
                    df, 
                    x="bond_return", 
                    nbins=50, 
                    title="Bond Return Distribution",
                    color_discrete_sequence=['#636EFA'],
                    marginal="box" 
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_dist2:
            # Correlation Heatmap for Return Columns
            return_cols = [c for c in df.columns if 'return' in c or 'yield' in c]
            if len(return_cols) > 0:
                corr = df[return_cols].corr()
                fig_corr = px.imshow(
                    corr, 
                    title="Correlation Matrix (Returns & Yields)",
                    color_continuous_scale='RdBu_r', 
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.subheader("Interactive Data Table")
        st.dataframe(df, width='stretch')

if __name__ == "__main__":
    main()
