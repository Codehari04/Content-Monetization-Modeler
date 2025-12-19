# ========================================================
# YOUTUBE MONETIZATION MODELER - COMPLETE APPLICATION
# ========================================================
# Author: Analytics Team
# Version: 2.0
# Description: Advanced revenue prediction & analytics platform
# ========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# ========================================================
# PAGE CONFIGURATION
# ========================================================
st.set_page_config(
    page_title="YouTube Monetization Modeler",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "YouTube Monetization Modeler v2.0 - Professional Analytics & Prediction Platform"
    }
)

# ========================================================
# CUSTOM STYLING & CSS
# ========================================================
st.markdown("""
    <style>
    /* Main container */
    .main {
        background-image: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
    }
    
    /* Header styling */
    .header-main {
        font-size: 2.8em;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
    }
    
    .header-sub {
        color: #475569;
        font-size: 1.15em;
        margin-top: 0;
    }
    
    /* Metric cards */
    .metric-card {
        color: #0f172a;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(8, 145, 178, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card-value {
        font-size: 1.8em;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #0f172a;
    }
    
    .metric-card-label {
        font-size: 0.9em;
        opacity: 0.95;
        margin-top: 5px;
        font-weight: 500;
    }
    
    /* Section divider */
    .section-divider {
        border: none;
        border-top: 2px solid #e2e8f0;
        margin: 30px 0;
    }
    
    /* Info boxes */
    .info-box {
        background-color: lightblue;
        border-left: 5px solid #0ea5e9;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    
    .success-box {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    
    .warning-box {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    
    /* Data table styling */
    .dataframe {
        font-size: 0.9em;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f1f5f9;
        border-radius: 6px;
        font-weight: 600;
        color: #334155;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
       background-image: linear-gradient(90deg, rgb(160, 222, 219),rgb(3, 165, 209));
        border-radius: 10px;
        margin-top: 40px;
        border: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================================
# UTILITY FUNCTIONS
# ========================================================

@st.cache_data
def load_and_preprocess_data(file):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    
    # Handle missing values
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    
    return df, num_cols, cat_cols

def engineer_features(df):
    """Create engineered features"""
    # df = df.copy() # Removed to save memory as st.cache_data returns a copy
    
    # Prevent division by zero
    df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"].replace(0, 1)
    df["watch_time_per_view"] = df["watch_time_minutes"] / df["views"].replace(0, 1)
    df["rpm"] = (df["ad_revenue_usd"] / df["views"].replace(0, 1)) * 1000
    df["like_comment_ratio"] = df["likes"] / (df["comments"].replace(0, 1))
    df["video_length_engagement"] = df["video_length_minutes"] * df["engagement_rate"]
    
    # Replace infinities
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

def format_currency(value):
    """Format as currency"""
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    return f"${value:.2f}"

def format_number(value):
    """Format large numbers"""
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:.0f}"

def create_download_button(data, filename, label, file_type="csv"):
    """Create download button"""
    if file_type == "csv":
        csv = data.to_csv(index=False)
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime="text/csv",
            use_container_width=True
        )

def get_percentile_rank(value, series):
    """Calculate percentile rank"""
    return (series <= value).sum() / len(series) * 100

# ========================================================
# HEADER
# ========================================================
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 class='header-main'>📊 YouTube Monetization Modeler</h1>
        <p class='header-sub'>AI-Powered Revenue Prediction & Advanced Analytics Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ========================================================
# DATA UPLOAD & PREPROCESSING
# ========================================================
with st.sidebar:
    st.header("🔧 Configuration & Settings")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📁 Upload YouTube Monetization Dataset (CSV)",
        type=["csv"],
        help="Dataset must contain: views, likes, comments, watch_time_minutes, video_length_minutes, subscribers, category, device, country, ad_revenue_usd"
    )

if uploaded_file is None:
    st.markdown("""
        <div class='info-box'>
            <h3>📁 Getting Started</h3>
            <p>Upload a CSV dataset to begin analysis. The dataset should contain the following columns:</p>
            <ul>
                <li><strong>Metrics:</strong> views, likes, comments, watch_time_minutes, video_length_minutes</li>
                <li><strong>Context:</strong> category, device, country, subscribers</li>
                <li><strong>Target:</strong> ad_revenue_usd</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load and preprocess data
df, num_cols, cat_cols = load_and_preprocess_data(uploaded_file)
df = engineer_features(df)
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# Validation
required_cols = {"views", "likes", "comments", "watch_time_minutes", "video_length_minutes", 
                 "subscribers", "category", "device", "country", "ad_revenue_usd"}

if not required_cols.issubset(df.columns):
    st.error(f"❌ Missing required columns: {required_cols - set(df.columns)}")
    st.stop()

# ========================================================
# SIDEBAR NAVIGATION & FILTERS
# ========================================================
with st.sidebar:
    st.markdown("---")
    st.header("📑 Navigation")
    page = st.radio(
        "Select Section",
        [
            "🏠 Executive Dashboard",
            "📈 Exploratory Data Analysis",
            "💡 Revenue Drivers",
            "🎯 Segment Performance",
            "🤖 Model Training & Diagnostics",
            "🔮 Revenue Predictor",
            "🔬 Advanced Analytics"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.header("🔍 Global Filters")
    
    min_views = st.slider("Minimum Views", int(df["views"].min()), int(df["views"].max()), 
                          int(df["views"].quantile(0.25)))
    
    category_filter = st.multiselect(
        "Categories",
        sorted(df["category"].unique()),
        default=sorted(df["category"].unique())
    )
    
    device_filter = st.multiselect(
        "Devices",
        sorted(df["device"].unique()),
        default=sorted(df["device"].unique())
    )
    
    country_filter = st.multiselect(
        "Countries",
        sorted(df["country"].unique())[:10],
        default=sorted(df["country"].unique())[:3]
    )
    
    # Apply filters
    df_filtered = df[
        (df["views"] >= min_views) &
        (df["category"].isin(category_filter)) &
        (df["device"].isin(device_filter)) &
        (df["country"].isin(country_filter))
    ]

# ========================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ========================================================
if page == "🏠 Executive Dashboard":
    st.subheader("📊 Executive Overview & KPIs")
    
    # Key Metrics Row 1
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_revenue = df_filtered["ad_revenue_usd"].sum()
    avg_revenue = df_filtered["ad_revenue_usd"].mean()
    avg_rpm = df_filtered["rpm"].mean()
    total_views = df_filtered["views"].sum()
    avg_engagement = (df_filtered["engagement_rate"].mean()) * 100
    
    with col1:
        st.metric("💰 Total Revenue", format_currency(total_revenue), 
                 f"{total_revenue:,.0f}")
    with col2:
        st.metric("📊 Avg Revenue/Video", format_currency(avg_revenue),
                 f"{avg_revenue:,.0f}")
    with col3:
        st.metric("📈 Avg RPM", format_currency(avg_rpm),
                 f"{avg_rpm:.2f}")
    with col4:
        st.metric("👁️ Total Views", format_number(total_views),
                 f"{total_views:,.0f}")
    with col5:
        st.metric("🎯 Avg Engagement %", f"{avg_engagement:.2f}%",
                 f"{df_filtered['engagement_rate'].mean():.4f}")
    
    st.markdown("---")
    
    # Revenue Distribution Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Revenue Distribution")
        fig_hist = px.histogram(
            df_filtered,
            x="ad_revenue_usd",
            nbins=50,
            title="Revenue Distribution (Histogram with KDE)",
            labels={"ad_revenue_usd": "Revenue ($)"},
            color_discrete_sequence=["#667eea"]
        )
        fig_hist.add_vline(
            x=df_filtered["ad_revenue_usd"].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text="Mean",
            annotation_position="top right"
        )
        fig_hist.add_vline(
            x=df_filtered["ad_revenue_usd"].median(),
            line_dash="dot",
            line_color="green",
            annotation_text="Median",
            annotation_position="top left"
        )
        fig_hist.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("### Views vs Revenue (Correlation)")
        fig_scatter = px.scatter(
            df_filtered,
            x="views",
            y="ad_revenue_usd",
            color="engagement_rate",
            size="subscribers",
            trendline="ols",
            hover_data={"views": ":,", "ad_revenue_usd": ":.2f", "engagement_rate": ":.3f"},
            title="Revenue vs Views (colored by Engagement)",
            labels={"views": "Views", "ad_revenue_usd": "Revenue ($)", "engagement_rate": "Engagement Rate"},
            color_continuous_scale="Viridis",
            opacity=0.6
        )
        fig_scatter.update_layout(height=450, hovermode="closest")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Time Series & Category Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Revenue by Category")
        category_data = df_filtered.groupby("category")["ad_revenue_usd"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=True).reset_index()
        
        fig_cat = px.bar(
            category_data,
            x="sum",
            y="category",
            orientation="h",
            title="Total Revenue by Category",
            labels={"sum": "Total Revenue ($)", "category": "Category"},
            color="sum",
            color_continuous_scale="Viridis",
            hover_data=["count"],
            text="sum"
        )
        fig_cat.update_traces(texttemplate="$%{text:.0f}", textposition="auto")
        fig_cat.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.markdown("### Revenue by Device")
        device_data = df_filtered.groupby("device")["ad_revenue_usd"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=True).reset_index()
        
        fig_dev = px.bar(
            device_data,
            x="sum",
            y="device",
            orientation="h",
            title="Total Revenue by Device",
            labels={"sum": "Total Revenue ($)", "device": "Device"},
            color="sum",
            color_continuous_scale="Plasma",
            hover_data=["count"],
            text="sum"
        )
        fig_dev.update_traces(texttemplate="$%{text:.0f}", textposition="auto")
        fig_dev.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dev, use_container_width=True)
    
    st.markdown("---")
    
    # Top Countries
    st.markdown("### Top 15 Countries by Average Revenue")
    top_countries = df_filtered.groupby("country")["ad_revenue_usd"].agg(["mean", "sum", "count"]).sort_values("mean", ascending=False).head(15).reset_index()
    
    fig_countries = px.bar(
        top_countries,
        x="mean",
        y="country",
        orientation="h",
        title="Top 15 Countries - Average Revenue per Video",
        labels={"mean": "Average Revenue ($)", "country": "Country"},
        color="mean",
        color_continuous_scale="RdYlGn",
        text="mean",
        hover_data=["count", "sum"]
    )
    fig_countries.update_traces(texttemplate="$%{text:.2f}", textposition="auto")
    fig_countries.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_countries, use_container_width=True)

# ========================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ========================================================
elif page == "📈 Exploratory Data Analysis":
    st.subheader("📊 Comprehensive Exploratory Data Analysis")
    
    # Dataset Overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📋 Total Records", f"{len(df_filtered):,}")
    with col2:
        st.metric("🔢 Numeric Features", len(num_cols))
    with col3:
        st.metric("🏷️ Categorical Features", len(cat_cols))
    with col4:
        st.metric("❌ Missing Values", df_filtered.isnull().sum().sum())
    with col5:
        st.metric("💾 Memory Usage", f"{df_filtered.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    st.markdown("---")
    
    # Correlation Heatmap
    st.markdown("### 🔥 Correlation Matrix Heatmap")
    
    correlation_cols = ["views", "likes", "comments", "watch_time_minutes", "video_length_minutes",
                        "subscribers", "engagement_rate", "rpm", "ad_revenue_usd"]
    corr_matrix = df_filtered[correlation_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu",
        text_auto=".2f",
        title="Feature Correlation Matrix",
        labels={"color": "Correlation Coefficient"},
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(height=700, width=800)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # Statistical Summary
    st.markdown("### 📊 Statistical Summary")
    
    stats_summary = df_filtered[num_cols].describe().T
    stats_summary["skewness"] = df_filtered[num_cols].skew()
    stats_summary["kurtosis"] = df_filtered[num_cols].kurtosis()
    
    st.dataframe(stats_summary.round(4), use_container_width=True)
    
    create_download_button(stats_summary.reset_index(), "statistics_summary.csv", 
                          "📥 Download Statistics")
    
    st.markdown("---")
    
    # Distribution Analysis
    st.markdown("### 📉 Distribution Analysis by Feature")
    
    selected_features = st.multiselect(
        "Select features to visualize",
        num_cols,
        default=["views", "ad_revenue_usd", "engagement_rate", "rpm"]
    )
    
    if selected_features:
        cols_per_row = 2
        rows = (len(selected_features) + cols_per_row - 1) // cols_per_row
        
        for i in range(rows):
            col1, col2 = st.columns(2)
            
            for j, col_idx in enumerate([i*cols_per_row, i*cols_per_row + 1]):
                if col_idx < len(selected_features):
                    feature = selected_features[col_idx]
                    
                    with col1 if j == 0 else col2:
                        fig_dist = px.histogram(
                            df_filtered,
                            x=feature,
                            nbins=40,
                            title=f"Distribution of {feature}",
                            labels={feature: feature},
                            color_discrete_sequence=["#667eea"]
                        )
                        fig_dist.update_layout(height=350, showlegend=False)
                        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.markdown("---")
    
    # Categorical Distribution
    st.markdown("### 🏷️ Categorical Features Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Categories")
        fig_cat_dist = px.bar(
            df_filtered["category"].value_counts().reset_index(),
            x="count",
            y="category",
            orientation="h",
            title="Video Distribution by Category",
            labels={"count": "Count", "category": "Category"},
            color="count",
            color_continuous_scale="Blues"
        )
        fig_cat_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_cat_dist, use_container_width=True)
    
    with col2:
        st.markdown("#### Devices")
        fig_dev_dist = px.bar(
            df_filtered["device"].value_counts().reset_index(),
            x="count",
            y="device",
            orientation="h",
            title="Video Distribution by Device",
            labels={"count": "Count", "device": "Device"},
            color="count",
            color_continuous_scale="Greens"
        )
        fig_dev_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dev_dist, use_container_width=True)
    
    st.markdown("---")
    
    # Data Quality Report
    st.markdown("### ✅ Data Quality Report")
    
    quality_report = {
        "Total Records": len(df_filtered),
        "Duplicate Records": len(df) - len(df.drop_duplicates()),
        "Missing Values": df_filtered.isnull().sum().sum(),
        "Complete Cases": len(df_filtered[df_filtered.isnull().sum(axis=1) == 0]),
        "Completeness %": (1 - df_filtered.isnull().sum().sum() / (len(df_filtered) * len(df_filtered.columns))) * 100
    }
    
    quality_df = pd.DataFrame(list(quality_report.items()), columns=["Metric", "Value"])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📋 Records", quality_report["Total Records"])
    with col2:
        st.metric("✨ Complete Cases", quality_report["Complete Cases"])
    with col3:
        st.metric("❌ Missing", quality_report["Missing Values"])
    with col4:
        st.metric("📊 Completeness", f"{quality_report['Completeness %']:.2f}%")
    with col5:
        st.metric("🔄 Duplicates", quality_report["Duplicate Records"])

# ========================================================
# PAGE 3: REVENUE DRIVERS
# ========================================================
elif page == "💡 Revenue Drivers":
    st.subheader("💡 Revenue Drivers Analysis")
    
    st.markdown("""
        <div class='info-box'>
            Understand which factors have the strongest impact on ad revenue generation.
            This analysis identifies key performance indicators that drive monetization.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Engagement Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Engagement Rate Impact")
        fig_eng = px.scatter(
            df_filtered,
            x="engagement_rate",
            y="ad_revenue_usd",
            trendline="ols",
            title="Engagement Rate vs Revenue",
            labels={"engagement_rate": "Engagement Rate", "ad_revenue_usd": "Ad Revenue ($)"},
            color="rpm",
            size="views",
            color_continuous_scale="Turbo",
            hover_data={"views": ":,", "ad_revenue_usd": ":.2f"},
            opacity=0.6
        )
        fig_eng.update_layout(height=450)
        st.plotly_chart(fig_eng, use_container_width=True)
    
    with col2:
        st.markdown("### ⏱️ Watch Time Impact")
        fig_wt = px.scatter(
            df_filtered,
            x="watch_time_minutes",
            y="ad_revenue_usd",
            trendline="ols",
            title="Watch Time vs Revenue",
            labels={"watch_time_minutes": "Watch Time (min)", "ad_revenue_usd": "Ad Revenue ($)"},
            color="views",
            size="engagement_rate",
            color_continuous_scale="Greens",
            hover_data={"views": ":,", "ad_revenue_usd": ":.2f"},
            opacity=0.6
        )
        fig_wt.update_layout(height=450)
        st.plotly_chart(fig_wt, use_container_width=True)
    
    st.markdown("---")
    
    # Subscribers Impact
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👥 Subscriber Impact")
        fig_subs = px.scatter(
            df_filtered,
            x="subscribers",
            y="ad_revenue_usd",
            trendline="ols",
            title="Subscribers vs Revenue",
            labels={"subscribers": "Subscribers", "ad_revenue_usd": "Ad Revenue ($)"},
            color="category",
            opacity=0.6
        )
        fig_subs.update_layout(height=450)
        st.plotly_chart(fig_subs, use_container_width=True)
    
    with col2:
        st.markdown("### 🎬 Video Length Impact")
        fig_len = px.scatter(
            df_filtered,
            x="video_length_minutes",
            y="ad_revenue_usd",
            trendline="ols",
            title="Video Length vs Revenue",
            labels={"video_length_minutes": "Video Length (min)", "ad_revenue_usd": "Ad Revenue ($)"},
            color="device",
            opacity=0.6
        )
        fig_len.update_layout(height=450)
        st.plotly_chart(fig_len, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance through Correlation
    st.markdown("### 🔝 Top Revenue Drivers (by Correlation)")
    
    correlation_cols = ["views", "likes", "comments", "watch_time_minutes", "video_length_minutes",
                        "subscribers", "engagement_rate", "rpm", "ad_revenue_usd"]
    
    correlation_with_revenue = df_filtered[correlation_cols].corr()["ad_revenue_usd"].sort_values(ascending=False)
    
    # Convert to DataFrame for Plotly
    corr_df = correlation_with_revenue[1:].reset_index()
    corr_df.columns = ["Feature", "Correlation"]

    fig_imp = px.bar(
        corr_df,
        x="Correlation",
        y="Feature",
        orientation="h",
        title="Feature Correlation with Revenue",
        labels={"Correlation": "Correlation Coefficient", "Feature": "Feature"},
        color="Correlation",
        color_continuous_scale="RdYlGn",
        text="Correlation"
    )
    fig_imp.update_traces(texttemplate="%.3f", textposition="auto")
    fig_imp.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("---")
    
    # Likes vs Comments
    st.markdown("### 👍 Likes vs Comments Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_likes = px.scatter(
            df_filtered,
            x="likes",
            y="ad_revenue_usd",
            trendline="ols",
            title="Likes vs Revenue",
            labels={"likes": "Likes", "ad_revenue_usd": "Revenue ($)"},
            color_discrete_sequence=["#667eea"],
            opacity=0.6
        )
        fig_likes.update_layout(height=400)
        st.plotly_chart(fig_likes, use_container_width=True)
    
    with col2:
        fig_comments = px.scatter(
            df_filtered,
            x="comments",
            y="ad_revenue_usd",
            trendline="ols",
            title="Comments vs Revenue",
            labels={"comments": "Comments", "ad_revenue_usd": "Revenue ($)"},
            color_discrete_sequence=["#764ba2"],
            opacity=0.6
        )
        fig_comments.update_layout(height=400)
        st.plotly_chart(fig_comments, use_container_width=True)

# ========================================================
# PAGE 4: SEGMENT PERFORMANCE
# ========================================================
elif page == "🎯 Segment Performance":
    st.subheader("🎯 Segment Performance Analysis")
    
    # Category × Device Heatmap
    st.markdown("### 🔥 Category × Device Revenue Heatmap")
    
    pivot_data = df_filtered.pivot_table(
        values="ad_revenue_usd",
        index="category",
        columns="device",
        aggfunc="mean"
    )
    
    fig_heatmap = px.imshow(
        pivot_data,
        labels={"x": "Device", "y": "Category", "color": "Avg Revenue ($)"},
        color_continuous_scale="RdYlGn",
        text_auto=".2f",
        title="Average Revenue: Category × Device Heatmap"
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # Box Plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 Revenue by Category (Box Plot)")
        fig_box_cat = px.box(
            df_filtered,
            x ="category",
y="ad_revenue_usd",
title="Revenue Distribution by Category",
        labels={"ad_revenue_usd": "Ad Revenue ($)"},
        color="category",
        points="outliers"
    )
    fig_box_cat.update_xaxes(tickangle=-45)
    fig_box_cat.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_box_cat, use_container_width=True)
    
    with col2:
        st.markdown("### 📦 Revenue by Device (Box Plot)")
        fig_box_dev = px.box(
            df_filtered,
            x="device",
            y="ad_revenue_usd",
            title="Revenue Distribution by Device",
            labels={"ad_revenue_usd": "Ad Revenue ($)"},
            color="device",
            points="outliers"
        )
        fig_box_dev.update_xaxes(tickangle=-45)
        fig_box_dev.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_box_dev, use_container_width=True)
    
    st.markdown("---")
    
    # Segment Statistics Table
    st.markdown("### 📊 Category Performance Statistics")
    
    category_stats = df_filtered.groupby("category").agg({
        "ad_revenue_usd": ["sum", "mean", "std", "min", "max", "count"],
        "views": "sum",
        "engagement_rate": "mean",
        "rpm": "mean"
    }).round(2)
    
    category_stats.columns = ["Total Revenue", "Avg Revenue", "Std Dev", "Min Revenue", 
                              "Max Revenue", "Video Count", "Total Views", "Avg Engagement", "Avg RPM"]
    
    st.dataframe(category_stats, use_container_width=True)
    create_download_button(category_stats.reset_index(), "category_statistics.csv",
                          "📥 Download Category Stats")
    
    st.markdown("---")
    
    # Device Performance
    st.markdown("### 📱 Device Performance Statistics")
    
    device_stats = df_filtered.groupby("device").agg({
        "ad_revenue_usd": ["sum", "mean", "std", "min", "max", "count"],
        "views": "sum",
        "engagement_rate": "mean",
        "rpm": "mean"
    }).round(2)
    
    device_stats.columns = ["Total Revenue", "Avg Revenue", "Std Dev", "Min Revenue",
                           "Max Revenue", "Video Count", "Total Views", "Avg Engagement", "Avg RPM"]
    
    st.dataframe(device_stats, use_container_width=True)
    create_download_button(device_stats.reset_index(), "device_statistics.csv", "📥 Download Device Stats")
    
    st.markdown("---")
    
    # Geographic Performance
    st.markdown("### 🌍 Geographic Performance")
    
    country_stats = df_filtered.groupby("country").agg({
        "ad_revenue_usd": ["sum", "mean", "count"],
        "views": "sum"
    }).sort_values(("ad_revenue_usd", "sum"), ascending=False).head(15)
    
    # Flatten MultiIndex columns
    country_stats.columns = ['_'.join(col).strip() for col in country_stats.columns.values]
    country_stats = country_stats.reset_index()
    
    fig_geo = px.bar(
        country_stats,
        x="ad_revenue_usd_mean",
        y="country",
        orientation="h",
        title="Top 15 Countries - Average Revenue",
        labels={"ad_revenue_usd_mean": "Average Revenue ($)", "country": "Country"},
        color="ad_revenue_usd_mean",
        color_continuous_scale="Viridis"
    )
    fig_geo.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_geo, use_container_width=True)
elif page == "🤖 Model Training & Diagnostics":
    st.subheader("🤖 Advanced Regression Modeling")
    st.markdown("""
        <div class='info-box'>
            Train and compare 5 different regression models to predict YouTube ad revenue.
            This page shows model performance, diagnostics, and feature importance analysis.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Data Preparation
    if len(df_filtered) < 10:
        st.error("❌ Not enough data points for training (minimum 10 required). Please adjust your filters in the sidebar.")
        st.stop()
        
    X = df_filtered.drop("ad_revenue_usd", axis=1)
    y = df_filtered["ad_revenue_usd"]
    
    num_features = X.select_dtypes(include=np.number).columns.tolist()
    cat_features = X.select_dtypes(include="object").columns.tolist()
    
    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ]
    )
    
    # Model Definitions
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=None),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    st.info("⏳ Training models... This may take a moment.")
    
    progress_bar = st.progress(0)
    results = []
    trained_pipes = {}
    best_pipe = None
    best_model_name = None
    best_r2 = -np.inf
    
    for idx, (name, model) in enumerate(models.items()):
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])
        
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)
        
        # Cross-validation
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
        
        results.append({
            "Model": name,
            "R² Score": r2,
            "RMSE ($)": rmse,
            "MAE ($)": mae,
            "MAPE (%)": mape * 100,
            "CV Mean R²": cv_scores.mean(),
            "CV Std": cv_scores.std()
        })
        
        trained_pipes[name] = pipe
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipe = pipe
        
        progress_bar.progress((idx + 1) / len(models))
    
    results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False)
    
    st.success("✅ Model training completed!")
    st.markdown("---")
    
    # Model Comparison
    st.markdown("### 🏆 Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_r2 = px.bar(
            results_df,
            x="Model",
            y="R² Score",
            color="R² Score",
            color_continuous_scale="Viridis",
            title="R² Score Comparison",
            labels={"R² Score": "R² Score"},
            text="R² Score",
            hover_data={"CV Mean R²": ":.4f"}
        )
        fig_r2.update_traces(texttemplate="%.4f", textposition="auto")
        fig_r2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        fig_rmse = px.bar(
            results_df,
            x="Model",
            y="RMSE ($)",
            color="RMSE ($)",
            color_continuous_scale="Reds_r",
            title="RMSE Comparison (Lower is Better)",
            labels={"RMSE ($)": "RMSE ($)"},
            text="RMSE ($)",
        )
        fig_rmse.update_traces(texttemplate="$%.2f", textposition="auto")
        fig_rmse.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed Metrics Table
    st.markdown("### 📊 Detailed Model Metrics")
    
    metrics_display = results_df.copy()
    st.dataframe(metrics_display, use_container_width=True)
    create_download_button(metrics_display, "model_results.csv", "📥 Download Model Results")
    
    st.markdown("---")
    
    # Best Model Announcement
    st.markdown(f"""
        <div class='success-box'>
            <h3>🏆 Best Performing Model: <strong>{best_model_name}</strong></h3>
            <p><strong>R² Score:</strong> {best_r2:.4f} | 
               <strong>RMSE:</strong> ${results_df.iloc[0]['RMSE ($)']:.2f} |
               <strong>MAE:</strong> ${results_df.iloc[0]['MAE ($)']:.2f}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipe, "models/best_model.pkl")
    st.success("✅ Best model saved to 'models/best_model.pkl'")
    
    st.markdown("---")
    
    # Feature Importance (for tree-based models)
    if hasattr(best_pipe.named_steps["model"], "feature_importances_"):
        st.markdown("### 🎯 Feature Importance Analysis")
        
        feature_names = num_features.copy()
        cat_encoder = best_pipe.named_steps["prep"].named_transformers_["cat"]
        cat_feature_names = cat_encoder.get_feature_names_out(cat_features).tolist()
        feature_names.extend(cat_feature_names)
        
        importances = best_pipe.named_steps["model"].feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=True).tail(20)
        
        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 20 Most Important Features",
            color="Importance",
            color_continuous_scale="Blues",
            text="Importance"
        )
        fig_importance.update_traces(texttemplate="%.4f", textposition="auto")
        fig_importance.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("---")
    
    # Residual Analysis
    st.markdown("### 📉 Residual Diagnostics")
    
    preds_best = best_pipe.predict(X_test)
    residuals = y_test - preds_best
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Residual", f"${residuals.mean():.2f}")
    with col2:
        st.metric("Std Residual", f"${residuals.std():.2f}")
    with col3:
        st.metric("Mean Abs Error", f"${np.abs(residuals).mean():.2f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_residual_dist = px.histogram(
            residuals,
            nbins=40,
            title="Residual Distribution",
            labels={"value": "Residual ($)"},
            color_discrete_sequence=["#667eea"]
        )
        fig_residual_dist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_residual_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_residual_dist, use_container_width=True)
    
    with col2:
        fig_qq = px.scatter(
            x=np.sort(residuals),
            y=np.sort(np.random.normal(0, residuals.std(), len(residuals))),
            title="Q-Q Plot (Normality Check)",
            labels={"x": "Actual Residuals", "y": "Theoretical Quantiles"},
            color_discrete_sequence=["#764ba2"],
            opacity=0.6
        )
        fig_qq.update_layout(height=400)
        st.plotly_chart(fig_qq, use_container_width=True)
    
    st.markdown("---")
    
    # Actual vs Predicted
    st.markdown("### 📊 Actual vs Predicted Values")
    
    pred_comparison = pd.DataFrame({
        "Actual": y_test,
        "Predicted": preds_best,
        "Error": residuals,
        "Error %": (np.abs(residuals) / y_test) * 100
    }).reset_index(drop=True)
    
    fig_pred_actual = px.scatter(
        x=y_test,
        y=preds_best,
        labels={"x": "Actual Revenue ($)", "y": "Predicted Revenue ($)"},
        title="Actual vs Predicted Revenue",
        color=np.abs(residuals),
        color_continuous_scale="Reds",
        opacity=0.6,
        hover_data={
            "x": ":.2f",
            "y": ":.2f",
            "color": ":.2f"
        }
    )
    
    fig_pred_actual.add_shape(
        type="line",
        x0=y_test.min(),
        x1=y_test.max(),
        y0=y_test.min(),
        y1=y_test.max(),
        line=dict(dash="dash", color="green", width=2),
    )
    
    fig_pred_actual.update_layout(height=450)
    st.plotly_chart(fig_pred_actual, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Sample Predictions")
    st.dataframe(pred_comparison.head(20), use_container_width=True)
    create_download_button(pred_comparison, "predictions.csv", "📥 Download All Predictions")
elif page == "🔮 Revenue Predictor":
    st.subheader("🔮 Interactive Revenue Forecasting Tool")
    st.markdown("""
        <div class='info-box'>
            Use the trained machine learning model to predict ad revenue for custom video scenarios.
            Adjust the inputs to see how different metrics affect revenue potential.
        </div>
    """, unsafe_allow_html=True)
    
    if not os.path.exists("models/best_model.pkl"):
        st.error("❌ No trained model found. Please train the model in 'Model Training' first.")
        st.stop()
    
    model = joblib.load("models/best_model.pkl")
    
    st.markdown("---")
    st.markdown("### 📝 Input Video Metrics")
    
    with st.form("prediction_form", border=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👀 Audience Metrics")
            views = st.number_input("Views", min_value=1, value=10000, step=1000, 
                                   help="Total video views")
            likes = st.number_input("Likes", min_value=0, value=500, step=50,
                                   help="Total likes received")
            comments = st.number_input("Comments", min_value=0, value=100, step=10,
                                      help="Total comments received")
            subscribers = st.number_input("Subscribers", min_value=0, value=100000, step=10000,
                                         help="Channel subscriber count")
        
        with col2:
            st.markdown("#### ⏱️ Content Metrics")
            watch_time = st.number_input("Watch Time (minutes)", min_value=1.0, value=50000.0, 
                                        step=5000.0, help="Total watch time in minutes")
            video_length = st.number_input("Video Length (minutes)", min_value=0.1, value=15.0, 
                                          step=0.5, help="Duration of the video")
        
        with col3:
            st.markdown("#### 🏷️ Contextual Info")
            category = st.selectbox("Category", sorted(df["category"].unique()),
                                   help="Content category")
            device = st.selectbox("Device", sorted(df["device"].unique()),
                                 help="Primary device type")
            country = st.selectbox("Country", sorted(df["country"].unique()),
                                  help="Primary audience country")
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Predict Revenue", use_container_width=True)
    
    if submitted:
        input_df = pd.DataFrame([{
            "views": views,
            "likes": likes,
            "comments": comments,
            "watch_time_minutes": watch_time,
            "video_length_minutes": video_length,
            "subscribers": subscribers,
            "category": category,
            "device": device,
            "country": country,
            "engagement_rate": (likes + comments) / max(views, 1),
            "watch_time_per_view": watch_time / max(views, 1),
            "rpm": 0,
            "like_comment_ratio": likes / (comments + 1),
            "video_length_engagement": video_length * ((likes + comments) / max(views, 1))
        }])
        
        prediction = model.predict(input_df)[0]
        est_rpm = (prediction / max(views, 1)) * 1000
        engagement_pct = ((likes + comments) / max(views, 1)) * 100
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Estimated Revenue", format_currency(prediction),
                     f"${prediction:,.2f}")
        with col2:
            st.metric("📈 Estimated RPM", format_currency(est_rpm),
                     f"${est_rpm:.2f}")
        with col3:
            st.metric("👁️ Views Input", format_number(views),
                     f"{views:,}")
        with col4:
            st.metric("🎯 Engagement %", f"{engagement_pct:.2f}%",
                     f"{(likes + comments):,} interactions")
        
        st.markdown("---")
        
        # Comparison with Historical Data
        st.markdown("### 📈 Historical Data Comparison")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_revenue = df["ad_revenue_usd"].mean()
            diff = prediction - avg_revenue
            pct_diff = (diff / avg_revenue) * 100
            st.metric("vs Dataset Average",
                     f"{pct_diff:+.1f}%",
                     f"${diff:+,.2f}")
        
        with col2:
            percentile = (df["ad_revenue_usd"] <= prediction).sum() / len(df) * 100
            st.metric("Percentile Rank",
                     f"{percentile:.1f}th",
                     f"Better than {percentile:.0f}% of videos")
        
        with col3:
            max_revenue = df["ad_revenue_usd"].max()
            pct_of_max = (prediction / max_revenue) * 100
            st.metric("% of Max Revenue",
                     f"{pct_of_max:.1f}%",
                     f"${max_revenue:,.2f} is max")
        
        with col4:
            similar_videos = len(df[(df["category"] == category) & (df["device"] == device)])
            similar_avg = df[(df["category"] == category) & (df["device"] == device)]["ad_revenue_usd"].mean()
            st.metric("vs Similar Category/Device",
                     f"{(prediction / similar_avg) * 100:.1f}%",
                     f"Avg: ${similar_avg:,.2f}")
        
        st.markdown("---")
        
        # Visualization
        st.markdown("### 📊 Revenue Distribution Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = px.histogram(
                df,
                x="ad_revenue_usd",
                nbins=50,
                title="Historical Revenue Distribution with Your Prediction",
                labels={"ad_revenue_usd": "Ad Revenue ($)"},
                color_discrete_sequence=["#667eea"]
            )
            
            fig_dist.add_vline(x=prediction, line_dash="dash", line_color="red",
                              annotation_text=f"Your Prediction: ${prediction:,.0f}",
                              annotation_position="top right")
            fig_dist.add_vline(x=df["ad_revenue_usd"].mean(), line_dash="dot", line_color="green",
                              annotation_text=f"Average: ${avg_revenue:,.0f}",
                              annotation_position="top left")
            
            fig_dist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Category comparison
            category_avg = df[df["category"] == category]["ad_revenue_usd"].mean()
            
            fig_cat_comp = px.bar(
                x=[prediction, category_avg, avg_revenue],
                y=["Your Prediction", f"{category} Average", "Overall Average"],
                orientation="h",
                title="Revenue Comparison",
                labels={"x": "Revenue ($)"},
                color=["#667eea", "#764ba2", "#e0e0e0"],
                text="x"
            )
            fig_cat_comp.update_traces(texttemplate="$%{text:.0f}", textposition="auto")
            fig_cat_comp.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_cat_comp, use_container_width=True)
        
        st.markdown("---")
        
        # Input Summary
        st.markdown("### 📝 Input Summary")
        
        input_summary = pd.DataFrame({
            "Parameter": ["Views", "Likes", "Comments", "Watch Time (min)", "Video Length (min)",
                         "Subscribers", "Category", "Device", "Country", "Engagement %"],
            "Value": [f"{views:,}", f"{likes:,}", f"{comments:,}", f"{watch_time:,.0f}",
                     f"{video_length:.1f}", f"{subscribers:,}", category, device, country,
                     f"{engagement_pct:.2f}%"]
        })
        
        st.dataframe(input_summary, use_container_width=True, hide_index=True)
        
        # Download prediction
        prediction_export = pd.DataFrame({
            "Metric": ["Predicted Revenue", "Estimated RPM", "Views", "Engagement Rate %",
                      "Percentile Rank", "Category", "Device", "Country"],
            "Value": [prediction, est_rpm, views, engagement_pct, percentile, category,
                     device, country]
        })
        create_download_button(prediction_export, "revenue_prediction.csv","📥 Download Prediction")

elif page == "🔬 Advanced Analytics":
    st.subheader("🔬 Advanced Analytics & Insights")
    st.markdown("### 📊 Advanced Analytics")
    
    st.markdown("""
        <div class='info-box'>
            Dive deep into comprehensive analytics with advanced visualizations and statistical insights.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Scatter Matrix / Pair Plot
    st.markdown("### 🔗 Multi-Dimensional Relationship Analysis")
    
    selected_features_advanced = st.multiselect(
        "Select features for advanced analysis",
        ["views", "likes", "comments", "watch_time_minutes", "engagement_rate", "rpm", "ad_revenue_usd"],
        default=["views", "engagement_rate", "rpm", "ad_revenue_usd"]
    )
    
    if len(selected_features_advanced) >= 2:
        # Create a scatter matrix manually
        fig_pairs = px.scatter_matrix(
            df_filtered[selected_features_advanced],
            title="Feature Relationship Matrix",
            dimensions=selected_features_advanced,
            color_discrete_sequence=["#667eea"],
            labels={col: col for col in selected_features_advanced}
        )
        fig_pairs.update_traces(showupperhalf=False)
        fig_pairs.update_layout(height=800)
        st.plotly_chart(fig_pairs, use_container_width=True)
    
    st.markdown("---")
    
    # Revenue Segmentation
    st.markdown("### 💰 Revenue Segmentation Analysis")
    
    # Create revenue segments
    # Create revenue segments
    try:
        df_filtered["Revenue_Segment"] = pd.qcut(df_filtered["ad_revenue_usd"], q=3, labels=["Low Revenue", "Medium Revenue", "High Revenue"])
    except ValueError:
        # Fallback for low variance data
        df_filtered["Revenue_Segment"] = "Standard"
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_dist = df_filtered["Revenue_Segment"].value_counts()
        fig_seg = px.pie(
            values=segment_dist.values,
            names=segment_dist.index,
            title="Video Distribution by Revenue Segment",
            color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"]
        )
        fig_seg.update_layout(height=400)
        st.plotly_chart(fig_seg, use_container_width=True)
    
    with col2:
        seg_stats = df_filtered.groupby("Revenue_Segment").agg({
            "engagement_rate": "mean",
            "rpm": "mean",
            "views": "mean"
        })
        
        fig_seg_stats = px.bar(
            seg_stats.reset_index(),
            x="Revenue_Segment",
            y=["engagement_rate", "rpm", "views"],
            barmode="group",
            title="Segment Characteristics",
            labels={"value": "Value", "variable": "Metric"},
            color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"]
        )
        fig_seg_stats.update_layout(height=400)
        st.plotly_chart(fig_seg_stats, use_container_width=True)
    
    st.markdown("---")
    
    # Revenue by Multiple Dimensions
    st.markdown("### 🌐 Multi-Dimensional Revenue Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category + Device
        cat_dev_revenue = df_filtered.groupby(["category", "device"])["ad_revenue_usd"].mean().reset_index()
        
        fig_cat_dev = px.scatter(
            cat_dev_revenue,
            x="category",
            y="device",
            size="ad_revenue_usd",
            color="ad_revenue_usd",
            title="Average Revenue: Category × Device",
            labels={"ad_revenue_usd": "Avg Revenue ($)"},
            color_continuous_scale="Viridis",
            hover_data={"ad_revenue_usd": ":.2f"}
        )
        fig_cat_dev.update_layout(height=400)
        st.plotly_chart(fig_cat_dev, use_container_width=True)
    
    with col2:
        # Views Segments + Category
        try:
            df_filtered["View_Segment"] = pd.qcut(df_filtered["views"], q=3, labels=["Low Views", "Medium Views", "High Views"])
        except ValueError:
            df_filtered["View_Segment"] = "Standard"
        
        view_cat_revenue = df_filtered.groupby(["View_Segment", "category"])["ad_revenue_usd"].mean().reset_index()
        
        fig_view_cat = px.bar(
            view_cat_revenue,
            x="category",
            y="ad_revenue_usd",
            color="View_Segment",
            barmode="group",
            title="Revenue by Category & View Segment",
            labels={"ad_revenue_usd": "Avg Revenue ($)"},
            color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"]
        )
        fig_view_cat.update_layout(height=400)
        st.plotly_chart(fig_view_cat, use_container_width=True)
    
    st.markdown("---")
    
    # Time-based Analysis (if date column exists)
    if "date" in df.columns:
        st.markdown("### 📅 Temporal Revenue Trends")
        
        df_filtered["date"] = pd.to_datetime(df_filtered["date"], errors='coerce')
        daily_revenue = df_filtered.groupby(df_filtered["date"].dt.date).agg({
            "ad_revenue_usd": ["sum", "mean", "count"],
            "views": "sum"
        }).reset_index()
        daily_revenue.columns = ["Date", "Total_Revenue", "Avg_Revenue", "Video_Count", "Total_Views"]
        
        fig_trend = px.line(
            daily_revenue,
            x="Date",
            y="Total_Revenue",
            title="Daily Revenue Trend",
            labels={"Total_Revenue": "Total Revenue ($)", "Date": "Date"},
            color_discrete_sequence=["#667eea"]
        )
        fig_trend.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("---")
    
    # Outlier Analysis
    st.markdown("### 🎯 Outlier Detection & Analysis")
    
    Q1 = df_filtered["ad_revenue_usd"].quantile(0.25)
    Q3 = df_filtered["ad_revenue_usd"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_filtered[(df_filtered["ad_revenue_usd"] > upper_bound) | 
                           (df_filtered["ad_revenue_usd"] < lower_bound)]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Videos", len(df_filtered))
    with col2:
        st.metric("⚠️ Outliers Detected", len(outliers))
    with col3:
        st.metric("📈 Outlier %", f"{(len(outliers)/len(df_filtered)*100):.2f}%")
    with col4:
        st.metric("🎯 Normal Range", f"${lower_bound:.2f} - ${upper_bound:.2f}")
    
    if len(outliers) > 0:
        fig_outliers = px.box(
            df_filtered,
            y="ad_revenue_usd",
            title="Outlier Detection (Box Plot)",
            labels={"ad_revenue_usd": "Revenue ($)"},
            color_discrete_sequence=["#667eea"]
        )
        fig_outliers.update_layout(height=400)
        st.plotly_chart(fig_outliers, use_container_width=True)
        
        st.markdown("### 🏆 Top 10 Outliers (High Revenue)")
        top_outliers = outliers.nlargest(10, "ad_revenue_usd")[
            ["views", "likes", "comments", "watch_time_minutes", "category", "device", "ad_revenue_usd"]
        ]
        st.dataframe(top_outliers, use_container_width=True)
    st.markdown("""
 <div class='footer'>
<div style='margin-bottom: 20px;'>
<h3 style='margin: 0;'>📊 YouTube Monetization Modeler v2.0</h3>
<p style='color: #666; margin: 10px 0 0 0;'>Professional Revenue Analytics & Prediction Platform</p>
</div>
</div>
<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; text-align: left;'>
        <div>
            <h4 style='margin-top: 0;'>📦 Features</h4>
            <ul style='margin: 0; padding: 0; list-style: none; font-size: 0.9em;'>
                <li>✅ EDA & Data Visualization</li>
                <li>✅ Revenue Driver Analysis</li>
                <li>✅ ML Model Training</li>
                <li>✅ Revenue Prediction</li>
            </ul>
        </div>
        <div>
            <h4 style='margin-top: 0;'>🔧 Tech Stack</h4>
            <ul style='margin: 0; padding: 0; list-style: none; font-size: 0.9em;'>
                <li>🐍 Python 3.8+</li>
                <li>🎯 Scikit-learn</li>
                <li>📊 Plotly & Pandas</li>
                <li>🚀 Streamlit</li>
            </ul>
        </div>
        <div>
            <h4 style='margin-top: 0;'>📚 Models</h4>
            <ul style='margin: 0; padding: 0; list-style: none; font-size: 0.9em;'>
                <li>📈 Linear Regression</li>
                <li>🔗 Ridge & Lasso</li>
                <li>🌳 Random Forest</li>
                <li>⚡ Gradient Boosting</li>
            </ul>
        </div>
        <div>
            <h4 style='margin-top: 0;'>📞 Support</h4>
            <ul style='margin: 0; padding: 0; list-style: none; font-size: 0.9em;'>
                <li>📧 <a href='hariharan22td0674@svcet.ac.in' style='color: #667eea; text-decoration: none;'>Contact</a></li>
                <li>📖 <a href='Update later ' style='color: #667eea; text-decoration: none;'>Documentation</a></li>
                <li>🐛 <a href='Update later' style='color: #667eea; text-decoration: none;'>Report Issues</a></li>
                <li>⭐ <a href='https://github.com/Codehari04' style='color: #667eea; text-decoration: none;'>GitHub</a></li>
            </ul>
        </div>
    </div>
    
   
""", unsafe_allow_html=True)