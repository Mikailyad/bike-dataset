import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import streamlit as st

# Set the page layout to wide
st.set_page_config(layout="wide")


# Load data
@st.cache_data
def load_data():
    day_df = pd.read_csv("data/day.csv")
    return day_df


day_df = load_data()

# Drop unnecessary columns
day_df = day_df.drop(columns=["instant", "dteday"])

# Convert categorical columns
category_columns = ["season", "weathersit", "holiday", "mnth", "workingday", "weekday"]
for column in category_columns:
    day_df[column] = day_df[column].astype("category")

# Rename categories for month
if day_df["mnth"].dtype.name == "category":
    day_df["mnth"] = day_df["mnth"].cat.rename_categories(
        {
            1: "Januari",
            2: "Februari",
            3: "Maret",
            4: "April",
            5: "Mei",
            6: "Juni",
            7: "Juli",
            8: "Agustus",
            9: "September",
            10: "Oktober",
            11: "November",
            12: "Desember",
        }
    )

# Rename year values
day_df["yr"] = day_df["yr"].replace([0, 1], ["2011", "2012"])

# Key statistics (total counts)
total_sharing_bike = day_df["cnt"].sum()
total_registered = day_df["registered"].sum()
total_casual = day_df["casual"].sum()

# Display metrics in three columns
col1, col2, col3 = st.columns(3)
col1.metric(label="Total Sharing Bike", value=f"{total_sharing_bike}")
col2.metric(label="Total Registered", value=f"{total_registered}")
col3.metric(label="Total Casual", value=f"{total_casual}")

# Sidebar with date range information
st.sidebar.subheader("🚲 Dashboard Analisis Peminjaman Sepeda")
st.sidebar.write("Muhammad Mikail Ziyad - ML Bangkit Academy")

# Display the data
st.subheader("Data Awal")
st.dataframe(day_df.head())

# Information about the dataset
st.subheader("Informasi Dataset")
st.write(day_df.info())
st.write("Jumlah Missing Values:")
st.write(day_df.isna().sum())
st.write("Jumlah Duplikasi:", day_df.duplicated().sum())
st.write("Statistik Deskriptif:")
st.dataframe(day_df.describe())

# Monthly bike counts chart
st.subheader("Jumlah Peminjaman Sepeda per Bulan")
monthly_counts = (
    day_df.groupby(["mnth", "yr"], observed=False)["cnt"].sum().unstack(fill_value=0)
)
st.line_chart(monthly_counts)

# Heatmap for average bike rentals per month
st.subheader("Rata-rata Peminjaman Sepeda per Bulan berdasarkan Tahun")
heatmap_data = day_df.pivot_table(
    values="cnt", index="yr", columns="mnth", aggfunc="mean", observed=False
)

plt.figure(figsize=(14, 8))
sn.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    cbar_kws={"label": "Rata-rata Peminjaman Sepeda"},
)
plt.title("Rata-rata Peminjaman Sepeda per Bulan berdasarkan Tahun")
plt.xlabel("Bulan")
plt.ylabel("Tahun")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
st.pyplot(plt)

# Rename season categories
if day_df["season"].dtype.name == "category":
    day_df["season"] = day_df["season"].cat.rename_categories(
        {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    )

# Group by season and calculate mean bike counts
grouped_by_season = day_df.groupby("season", observed=False)["cnt"].mean().reset_index()

# Pie chart for average rentals per season
st.subheader("Proporsi Rata-rata Peminjaman Sepeda per Musim")
plt.figure(figsize=(8, 8))
plt.pie(
    grouped_by_season["cnt"],
    labels=grouped_by_season["season"],
    autopct=lambda p: "{:.1f}%".format(p) if p > 0 else "",
    startangle=90,
    colors=sn.color_palette("coolwarm", len(grouped_by_season)),
)

plt.title("Proporsi Rata-rata Peminjaman Sepeda per Musim")
plt.axis("equal")
st.pyplot(plt)
