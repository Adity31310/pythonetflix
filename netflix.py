import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_distplot
from collections import Counter

# Load dataset
path = r"C:/Users/adity/OneDrive/Desktop/netflix_titles.csv"
df = pd.read_csv(path)

# Data cleaning
df.dropna(subset=["date_added"], inplace=True)
df['director'].fillna("unknown", inplace=True)
df['cast'].fillna("unknown", inplace=True)
df['country'].fillna("unknown", inplace=True)
df.drop(columns='show_id', inplace=True)
df['date_added'] = df['date_added'].str.strip()
df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y')
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

# Content type pie chart
plt.pie(df.type.value_counts(), labels=df.type.value_counts().index, autopct="%.2f%%",
        textprops={'fontsize': 16}, colors=['lightsteelblue', 'lightsalmon'])
plt.title("Content Type Distribution")
plt.legend()
plt.show()

# Titles added per year - line and bar chart
title_counts = df.groupby('year_added')['title'].count()
plt.figure(figsize=(10, 6))
title_counts.plot(kind='line')
plt.title("Number of Titles Added to Netflix Over the Years")
plt.xlabel('Year')
plt.ylabel('Number of Content')
plt.show()

plt.figure(figsize=(10, 6))
title_counts.plot(kind='bar', color='green')
plt.title("Number of Titles Added to Netflix Over the Years")
plt.xlabel('Year')
plt.ylabel('Number of Content')
plt.show()

# Histogram grouped by content type
fig = px.histogram(df, x="year_added", color="type", barmode="group",
                   title="Content added over the years")
fig.show()

# Separate TV shows and Movies
d1 = df[df["type"] == "TV Show"]
d2 = df[df["type"] == "Movie"]

vc1 = d1["year_added"].value_counts().reset_index()
vc1.columns = ['year_added', 'count_tv_shows']
vc2 = d2["year_added"].value_counts().reset_index()
vc2.columns = ['year_added', 'count_movies']

# Line plot for TV shows and movies over years
fig = px.line(vc1.sort_values('year_added'), x="year_added", y="count_tv_shows", markers=True,
              title="Content added over the years (TV Shows vs Movies)")
fig.add_scatter(x=vc2.sort_values('year_added')['year_added'],
                y=vc2.sort_values('year_added')['count_movies'],
                mode="lines+markers", name="Movies", line=dict(color="green"))
fig.show()

# Monthly additions
month_count = df["month_added"].value_counts().sort_index().reset_index()
month_count.columns = ['month_added', 'count']
months = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]
plt.figure(figsize=(10, 6))
plt.bar(month_count["month_added"], month_count["count"])
plt.xticks(month_count["month_added"], months)
plt.xlabel("Month")
plt.ylabel("Number of contents")
plt.title("Monthly Content Additions")
plt.show()

# Country-wise top 10 content
country_data = df['country'].value_counts().head(10).reset_index()
country_data.columns = ['country', 'count']
labels = country_data['country'][::-1]
values = country_data['count'][::-1]

fig = go.Figure(data=[go.Bar(y=labels, x=values, orientation="h", marker=dict(color="#a678de"))])
fig.update_layout(title="Countries with the Most Content", height=700)
fig.show()



# Season counts for TV shows
df['season_count'] = df['duration'].where(df['type'] == 'TV Show', np.nan)
df['season_count'] = df['season_count'].str.extract('(\d+)').fillna(0).astype(int)
seasons = df[df['type'] == 'TV Show']['season_count'].value_counts().reset_index()
seasons.columns = ['season_count', 'count']
plt.bar(seasons['season_count'], seasons['count'])
plt.xlabel("Number of Seasons")
plt.ylabel("Number of Contents")
plt.title("Distribution of Seasons in TV Shows")
plt.show()

# Ratings distribution
overall = df['rating'].value_counts(normalize=True)
movie = df[df['type'] == 'Movie']['rating'].value_counts(normalize=True)
tv_show = df[df['type'] == 'TV Show']['rating'].value_counts(normalize=True)

ratings_df = pd.DataFrame({
    'Rating': overall.index,
    'Overall': overall.values,
    'Movies': movie.reindex(overall.index).fillna(0).values,
    'TV Shows': tv_show.reindex(overall.index).fillna(0).values
})
ratings_melted = ratings_df.melt(id_vars='Rating', var_name='Type', value_name='Percentage')
fig = px.bar(ratings_melted, x='Rating', y='Percentage', color='Type', barmode='group',
             title='Distribution of Content Ratings on Netflix')
fig.show()

# Genre distribution for movies
genres = ", ".join(d2['listed_in']).split(", ")
counter_list = Counter(genres).most_common(50)
labels = [_[0] for _ in counter_list][::-1]
values = [_[1] for _ in counter_list][::-1]
plt.figure(figsize=(10, 8))
plt.barh(labels, values, color="#a678de")
plt.xlabel('Frequency')
plt.ylabel('Categories')
plt.title('Top Genres in Movies')
plt.gca().invert_yaxis()
plt.show()
