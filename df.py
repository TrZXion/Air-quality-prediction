import sqlite3, csv
import pandas as pd
import numpy as np

df = pd.read_csv("air_quality_who.csv", encoding="Windows-1252")
df.columns = df.columns.str.strip()

connection = sqlite3.connect("air_quality.db")

df.to_sql("air_quality_dataset", connection, if_exists="replace")
connection.close()
