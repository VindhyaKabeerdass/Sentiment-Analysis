# Social-Media-Analysis-Twitter


Objective:

The aim of this project is to predict the political ideology (left wing or right wing) of a person through one’s music preferences

Data Sources:

The official Spotify playlists of 27 politicians from USA is considered for training the model. The dataset had a total of 1452 songs from the playlist of 17 Democrats and 10 Republicans.

Methodology:

Using Spotify’s API, the following features for each song are extracted

Acousticness
Danceability
Duration
Energy
Instrumentalness
Key
Liveness
Loudness
Mode
Speechiness
Tempo
Time signature
Valence
A logistic regression model is built using the above-mentioned parameters to predict the political affiliation of a person.

Conclusion:

The logistic regression model has made predictions with an accuracy of 81.10%
