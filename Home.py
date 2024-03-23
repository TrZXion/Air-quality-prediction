import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import umap.umap_ as umap


st.set_page_config(page_title="Dashboard", page_icon="📑", layout="wide")
st.subheader("🗺 Air Quality Analytics")
st.markdown("##")


df = pd.read_csv("air_quality_who.csv", encoding="Windows 1252")
df_clean = pd.read_csv("encoded_data.csv", encoding="Windows 1252")


# Side Panel
st.sidebar.image("data/sidebar_1.png", caption="Data Analysis")


# Switcher
st.sidebar.header("Data Filter")
Country = st.sidebar.multiselect(
    "Select Country",
    options=df["country_name"].unique(),
    default="India",
)

Year = st.sidebar.multiselect(
    "Select Year",
    options=df["year"].unique(),
    default={2016, 2017, 2018, 2019, 2020},
)


df_selection = df.query("country_name==@Country and year== @Year")


def Home():
    with st.expander("Raw Data"):
        showData = st.multiselect(
            "Filter:",
            df_selection.columns,
            default=[
                "country_name",
                "iso3",
                "pm10_concentration",
                "pm25_concentration",
                "no2_concentration",
                "pm25_tempcov",
            ],
        )
        st.write(df_selection[showData])


Home()

st.subheader("Cleaned Data")
st.write(df_clean)


# target_column = st.sidebar.selectbox("Select the target column", df_clean.columns)
target_column = "pm25_concentration"


@st.cache_data
def generate_pca_plot(df_clean, target_column):
    numeric_df = df_clean.select_dtypes(include="number")
    numeric_df.fillna(0, inplace=True)
    pca = PCA()
    pca.fit(df_clean)

    cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative_variance_ratio) + 1)),
            y=cumulative_variance_ratio,
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title="PCA - Cumulative Variance Explained",
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Variance Explained",
    )

    st.plotly_chart(fig, use_container_width=True)


generate_pca_plot(df_clean, target_column)


@st.cache_data
def generate_heatmap(df_clean):
    numeric_df = df_clean.select_dtypes(include="number")  # Select only numeric columns
    corr = numeric_df.corr()
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        showscale=True,
    )

    fig.update_layout(width=1000, height=1000)
    st.plotly_chart(fig)


generate_heatmap(df_clean)


@st.cache_data
def generate_distribution_plot(df_clean):
    numeric_cols = [col for col in df_clean.columns if df_clean[col].dtype != "object"]
    n = len(numeric_cols) // 4 + 1
    fig = make_subplots(rows=4, cols=n)

    for i, col in enumerate(numeric_cols):
        histogram = go.Histogram(x=df_clean[col], name=col)
        fig.add_trace(histogram, row=i // n + 1, col=i % n + 1)

    fig.update_layout(height=1000, width=1500)
    st.plotly_chart(fig)


st.subheader("Distribution Plot")
generate_distribution_plot(df_clean)

# TSNE


@st.cache_data
def generate_2d_tsne(df_clean, target_column):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(df_clean)

    df_tsne = pd.DataFrame(X_tsne, columns=["x", "y"])
    df_tsne["target"] = df_clean[target_column].values
    fig = px.scatter(
        df_tsne, x="x", y="y", color="target", color_continuous_scale="Blues"
    )
    fig.update_layout(title="2D TSNE")
    st.plotly_chart(fig, use_container_width=True)


generate_2d_tsne(df_clean, target_column)


@st.cache_data
def generate_2d_umap(df_clean, target_column):
    umap_ = umap.UMAP(n_components=2)
    X_umap = umap_.fit_transform(df_clean)

    df_umap = pd.DataFrame(X_umap, columns=["x", "y"])
    df_umap["target"] = df_clean[target_column].values
    fig = px.scatter(
        df_umap, x="x", y="y", color="target", color_continuous_scale="Blues"
    )
    fig.update_layout(title="2D UMAP")
    st.plotly_chart(fig, use_container_width=True)


generate_2d_umap(df_clean, target_column)


@st.cache_data
def generate_3d_tsne(df_clean, target_column):
    tsne = TSNE(n_components=3)
    X_tsne = tsne.fit_transform(df_clean)

    df_tsne = pd.DataFrame(X_tsne, columns=["x", "y", "z"])
    df_tsne["target"] = df_clean[target_column].values
    fig = px.scatter_3d(
        df_tsne, x="x", y="y", z="z", color="target", color_continuous_scale="Blues"
    )
    fig.update_layout(title="3D TSNE")
    fig.update_layout(width=1000, height=1000)
    st.plotly_chart(fig, use_container_width=True)


generate_3d_tsne(df_clean, target_column)


@st.cache_data
def generate_3d_umap(df_clean, target_column):
    umap_ = umap.UMAP(n_components=3)
    X_umap = umap_.fit_transform(df_clean)

    df_umap = pd.DataFrame(X_umap, columns=["x", "y", "z"])
    df_umap["target"] = df_clean[target_column].values
    fig = px.scatter_3d(
        df_umap, x="x", y="y", z="z", color="target", color_continuous_scale="Blues"
    )
    fig.update_layout(title="3D UMAP")
    fig.update_layout(width=1000, height=1000)
    st.plotly_chart(fig, use_container_width=True)


generate_3d_umap(df_clean, target_column)
