from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')
from flask_cors import CORS
import requests
app = Flask(__name__)
CORS(app)

# Load pickled data
# try:
movie_list = pickle.load(open('C:/Users/arun9/ReactProjects/MovieRecommnederApp/backend/artifacts/movie_list.pkl', 'rb'))
vector=cv.fit_transform(movie_list['tags']).toarray()
# similarity = pickle.load(open('C:/Users/arun9/ReactProjects/MovieRecommnederApp/backend/artifacts/similarity.pkl', 'rb'))
similarity= cosine_similarity(vector)
# similarity=similarity.values
# print(similarity.dtypes)
# print(similarity.head())
# if not np.issubdtype(similarity.dtype, np.number):
#     raise ValueError("Similarity matrix contains non-numeric values!")
print(f"Similarity Matrix Shape: {similarity.shape}, Type: {type(similarity)}")
print(f"Movie List Shape: {movie_list.shape[0]}, Similarity Matrix Rows: {similarity.shape[0]}")

# except FileNotFoundError as e:
#     print(f"Error: {e}")
#     exit()


def fetch_poster(movie_id):
    url="https://api.themoviedb.org/3/movie/{}?api_key=dc3e1a0b573c11ca04f5cc371062d4b0&&language=en-US".format(movie_id)
    data=requests.get(url)
    data=data.json()
    poster_path=data['poster_path']
    full_path="https://image.tmdb.org/t/p/w500" + poster_path

    return full_path
    



# Recommendation function
def recommend(movie):
    # Check if the movie exists in the dataset
    matching_movies = movie_list[movie_list['title'].str.lower() == movie.lower()]
    print(f"Matching Movies: {matching_movies}")
    if matching_movies.empty:
        return [f"Movie '{movie}' not found in the database."]
    
    # print(f"Movie Title: {movie}")
    # print(f"Index Found: {matching_movies.index[0]}")
    # print(f"Similarity Matrix Shape: {similarity.shape}")
    # print(f"Movie List Shape: {movie_list.shape}")

    try:
        print(f"Movie Title: {movie}")
        print(f"Index Found: {matching_movies.index[0]}")
        print(f"Similarity Matrix Shape: {similarity.shape}")
        print(f"Type of similarity: {type(similarity)}")
        print(f"Movie List Shape: {movie_list.shape}")

        # Get the index of the movie
        index = int(matching_movies.index[0])  # Ensure index is an integer
        print(index)
        
        # Check alignment
        if index >= similarity.shape[0]:
            return [f"Index out of bounds for similarity matrix. Index: {index}"]
       
        # print(f"Distances Before Sorting: {similarity[index]}")
        

        # Calculate similarity and sort
        distances = sorted(
            list(enumerate(similarity[index])),
            reverse=True,
            key=lambda x: x[1]
        )
        print(movie_list)
        recommended_movies_name=[]
        recommended_movies_poster=[]
        recommended_movies_id=[]
        
        # Fetch top 5 recommended movie titles
       
        for i in distances[1:21]:
             movie_id = int(movie_list.iloc[i[0]].movie_id)
             print(movie_id)
             recommended_movies_poster.append(fetch_poster(movie_id))
             recommended_movies_name.append(movie_list.iloc[i[0]].title)
             recommended_movies_id.append(movie_id)
        # print(f"dhb:{recommended_movies}")
        return recommended_movies_name,recommended_movies_poster,recommended_movies_id
    except Exception as e:
        return [f"An error occurred: {str(e)}"]
        
# l=[]
# l=recommend('spectre')
# print(l)

# API route
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    if not data or 'movie' not in data:
        return jsonify({'error': 'Please provide a valid movie name in JSON format: {"movie": "<movie_name>"}'}), 400
    
    movie_name = data.get('movie', '').strip()
    if not movie_name:
        return jsonify({'error': 'Movie name cannot be empty!'}), 400

    recommended_movies_name, recommended_movies_poster,recommended_movies_id = recommend(movie_name)
    return jsonify({
        'recommendations': recommended_movies_name,
        'posters': recommended_movies_poster,
        'id':recommended_movies_id
        })

if __name__ == '__main__':
    app.run(debug=True)
