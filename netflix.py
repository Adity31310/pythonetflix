import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv(r"C:\Users\abhis\OneDrive\Desktop\netflix_titles.csv")

df.head()

df.shape

df.dtypes

df.columns

df.info()

df.describe()

df.isnull().sum()

df.isnull().sum().sum()

df.director.fillna(value="unknown", inplace=True)

df.cast.fillna(value="unknown", inplace=True)

df.country.fillna(value="unknown", inplace =True)

df.dropna(inplace=True)

df.isnull().sum()

df.drop(columns="show_id", inplace=True)

df.head()

df["date_added"]

has_whitespaces = df["date_added"].str.startswith(" ")
indexes_with_whitespace = df[has_whitespaces].index

print("Indexes of lines containing spaces:")
print(indexes_with_whitespace)

df["date_added"] = df["date_added"].str.strip()

df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y')

df["date_added"]

df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

df.head(1)

df.head()

df["type"].unique()

df["type"].value_counts()

plt.pie(df.type.value_counts(), 
        labels = df.type.value_counts().index, 
        labeldistance = None, autopct="%.2f", 
        textprops = {'fontsize': 16,}, 
        colors = ['lightsteelblue','lightsalmon' ] )
plt.legend()
plt.show()

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

title_counts

d1 = df[df["type"] == "TV Show"]
d2 = df[df["type"] == "Movie"]


fig = px.histogram(df, x="year_added", color="type", barmode="group", title="Content added over the years")

fig.show()

d1 = df[df["type"] == "TV Show"]
d2 = df[df["type"] == "Movie"]

vc1 = d1["year_added"].value_counts().reset_index()
vc1 = vc1.rename(columns={"year_added": "year_added", "index": "count_tv_shows"})

vc2 = d2["year_added"].value_counts().reset_index()
vc2 = vc2.rename(columns={"year_added": "year_added", "index": "count_movies"})


fig = px.line(vc1, x="year_added", y="count", title="Content added over the years (TV Shows)",markers=True)
fig.add_scatter(x=vc2["year_added"], y=vc2["count"], mode="lines", name="Movies", line=dict(color="green"))


fig.show()

month_count = df["month_added"].value_counts().reset_index()
month_count = month_count.sort_values(by="month_added")  
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

plt.figure(figsize=(10,6))
plt.bar(month_count["month_added"], month_count["count"])
plt.xticks(month_count["month_added"], months)  
plt.xlabel("Oh")
plt.ylabel("Number of contents")
plt.show()

from plotly.offline import init_notebook_mode, iplot

labels = [_[0] for _ in tabs][::-1]
values = [_[1] for _ in tabs][::-1]
trace1 = go.Bar(y=labels, x=values, orientation="h", name="", marker=dict(color="#a678de"))

data = [trace1]
layout = go.Layout(title="Countries with the Most Content", height=700, legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
fig.show()

plt.barh(y=labels, width=values, color="#a678de", height=0.8)
plt.title("Countries with the Most Content")
plt.show()

import plotly.figure_factory as ff
x1 = d2['duration'].fillna(0.0).astype(float)
fig = ff.create_distplot([x1], ['a'], bin_size=0.7, curve_type='normal', colors=["#6ad49b"])
fig.update_layout(title_text='Distribution of Film Duration')
fig.show()

x1 = d2['duration'].fillna(0.0).astype(float)
sns.histplot(x=x1, kde=True, color="#6ad49b")

plt.title('Distribution of Film Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.show()

df.head()

df.loc[0, "season_count"]

seasons = df[df["season_count"] != ""]
seasons = seasons["season_count"].value_counts().reset_index()
seasons

plt.bar(seasons["season_count"], seasons["count"])
plt.xlabel("Number of Seasons")
plt.ylabel("Number of Contents")
plt.show()

overall_ratings_distribution = df['rating'].value_counts(normalize=True)

# Ratings distribution for movies and TV shows
movie_ratings_distribution = df[df['type'] == 'Movie']['rating'].value_counts(normalize=True)
tv_show_ratings_distribution = df[df['type'] == 'TV Show']['rating'].value_counts(normalize=True)

# Combine movie and TV show ratings into a single DataFrame
ratings_df = pd.DataFrame({
    'Rating': overall_ratings_distribution.index,
    'Overall': overall_ratings_distribution.values,
    'Movies': movie_ratings_distribution.reindex(overall_ratings_distribution.index).fillna(0).values,
    'TV Shows': tv_show_ratings_distribution.reindex(overall_ratings_distribution.index).fillna(0).values
})

# Melt the DataFrame for easier plotting
ratings_melted = ratings_df.melt(id_vars='Rating', var_name='Type', value_name='Percentage')

# Plotting using Plotly Express
fig = px.bar(ratings_melted, x='Rating', y='Percentage', color='Type', barmode='group',
            labels={'Percentage': 'Percentage of Total', 'Rating': 'Rating'},
            title='Distribution of Content Ratings on Netflix')
fig.show()

overall_ratings_distribution = df['rating'].value_counts(normalize=True)
ratings = overall_ratings_distribution.index
overall_percentages = overall_ratings_distribution.values * 100

# Ratings distribution for movies
movie_ratings_distribution = df[df['type'] == 'Movie']['rating'].value_counts(normalize=True)
movie_percentages = movie_ratings_distribution.reindex(ratings).fillna(0).values * 100

# Ratings distribution for TV shows
tv_show_ratings_distribution = df[df['type'] == 'TV Show']['rating'].value_counts(normalize=True)
tv_show_percentages = tv_show_ratings_distribution.reindex(ratings).fillna(0).values * 100

# Plotting
plt.figure(figsize=(12, 6))
bar_width = 0.3
index = range(len(ratings))

plt.bar(index, overall_percentages, bar_width, label='Overall', color='blue', alpha=0.7)
plt.bar([i + bar_width for i in index], movie_percentages, bar_width, label='Movies', color='red', alpha=0.7)
plt.bar([i + 2 * bar_width for i in index], tv_show_percentages, bar_width, label='TV Shows', color='green', alpha=0.7)

plt.xlabel('Rating')
plt.ylabel('Percentage of Total')
plt.title('Distribution of Content Ratings on Netflix')
plt.xticks([i + bar_width for i in index], ratings, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

genre_counts = df["listed_in"].value_counts()
genre_counts.head(10)

from collections import Counter


categories = ", ".join(d2['listed_in']).split(", ")
counter_list = Counter(categories).most_common(50)
labels = [_[0] for _ in counter_list][::-1]
values = [_[1] for _ in counter_list][::-1]


plt.figure(figsize=(10, 8))
plt.barh(labels, values, color="#a678de")
plt.xlabel('Frequency')
plt.ylabel('Categories')
plt.title('Content added over the years')
plt.gca().invert_yaxis()  
plt.show()

def country_trace(country, flag="movie"):
    df["from_us"] = df['country'].fillna("").apply(lambda x: 1 if country.lower() in x.lower() else 0)
    small = df[df["from_us"] == 1]
    if flag == "movie":
        small = small[small["duration"] != ""]
    else:
        small = small[small["season_count"] != ""]
    cast = ", ".join(small['cast'].fillna("")).split(", ")
    tags = Counter(cast).most_common(25)
    tags = [_ for _ in tags if _[0] != "unknown"]

    labels, values = [_[0] + "  " for _ in tags], [_[1] for _ in tags]
    trace = go.Bar(y=labels[::-1], x=values[::-1], orientation="h", name="")
    return trace

traces = []
titles = ["United States", "", "India", "", "United Kingdom", "", "Canada", "", "Spain", "", "Japan"]
valid_titles = [title for title in titles if title]  # Boş olmayan başlıkları seçin
for title in valid_titles:
    traces.append(country_trace(title))

fig = make_subplots(rows=2, cols=3, subplot_titles=valid_titles)
for i, trace in enumerate(traces, start=1):
    row = (i - 1) // 3 + 1
    col = (i - 1) % 3 + 1
    fig.add_trace(trace, row, col)

fig.update_layout(height=800, showlegend=False)
fig.show()

import matplotlib.pyplot as plt
from collections import Counter

def country_trace(country, flag="movie"):
    df["from_us"] = df['country'].fillna("").apply(lambda x: 1 if country.lower() in x.lower() else 0)
    small = df[df["from_us"] == 1]
    if flag == "movie":
        small = small[small["duration"] != ""]
    else:
        small = small[small["season_count"] != ""]
    cast = ", ".join(small['cast'].fillna("")).split(", ")
 
    cast = [_ for _ in cast if _ != "unknown"]
    tags = Counter(cast).most_common(25)

    labels, values = [_[0] + "  " for _ in tags], [_[1] for _ in tags]
    return labels[::-1], values[::-1]

titles = ["United States", "", "India", "", "United Kingdom", "", "Canada", "", "Spain", "", "Japan"]
valid_titles = [title for title in titles if title]  
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

for i, title in enumerate(valid_titles):
    labels, values = country_trace(title)
    ax = axs[i // 3, i % 3] 
    ax.barh(labels, values, color="#a678de")
    ax.set_title(title)

plt.tight_layout()
plt.show()

traces = []
titles = ["United States","", "United Kingdom"]
for title in titles:
    if title != "":
        traces.append(country_trace(title, flag="tv_shows"))

fig = make_subplots(rows=1, cols=3, subplot_titles=titles)
fig.add_trace(traces[0], 1,1)
fig.add_trace(traces[1], 1,3)

fig.update_layout(height=600, showlegend=False)
fig.show()

def country_trace(country, flag="movie"):
    df["from_us"] = df['country'].fillna("").apply(lambda x: 1 if country.lower() in x.lower() else 0)
    small = df[df["from_us"] == 1]
    if flag == "movie":
        small = small[small["duration"] != ""]
    else:
        small = small[small["season_count"] != ""]
    cast = ", ".join(small['cast'].fillna("")).split(", ")
    tags = Counter(cast).most_common(25)
    tags = [_ for _ in tags if _[0] != "unknown"]

    labels, values = [_[0] + "  " for _ in tags], [_[1] for _ in tags]
    return labels[::-1], values[::-1]

traces = []
titles = ["United States", "United Kingdom"]
for title in titles:
    labels, values = country_trace(title, flag="tv_shows")
    traces.append((labels, values))

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for i, (title, (labels, values)) in enumerate(zip(titles, traces)):
    axs[i].barh(labels, values, color="#a678de")
    axs[i].set_title(title)

plt.tight_layout()
plt.show()
