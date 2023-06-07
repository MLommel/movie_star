# import std libraries
import numpy as np
import pandas as pd
import time

from IPython.display import HTML
import pickle
import json
#from scipy.sparse import csr_matrix

#import models
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF 
from sklearn.neighbors import KNeighborsClassifier



import streamlit as st
from st_aggrid import AgGrid

movies = pd.read_csv('data/movies.csv')
Rt = pd.read_csv('my_Rt.csv')
#rating = pd.read_csv('data/ratings.csv')
#tags = pd.read_csv('data/tags.csv')
#links = pd.read_csv('data/links.csv')

# calculate the number of ratings per movie
#rating_count = rating.groupby('movieId')[['rating']].count()

# filter for movies with more than 20 ratings and extract the index
#popular_movies = rating_count[rating_count['rating']>20].index

# filter the ratings matrix and only keep the popular movies
#dfn = rating[rating['movieId'].isin(popular_movies)].copy()

# need to remake user ids and movie ids since they are not sequential
#user_ids = dfn['userId'].unique()
#user_id_map = {v:k for k,v in enumerate(user_ids)} #reduce with -1 user Id
#dfn['userId'] = dfn['userId'].map(user_id_map)


#movie_ids = dfn['movieId'].unique()
#movie_id_map = {v:k for k,v in enumerate(movie_ids)} 
#dfn['movieId'] = dfn['movieId'].map(movie_id_map)

# Initialize a sparse user-item rating matrix
# (data, (row_ind, col_ind) from scipy.sparse import csr_matrix
#careful csr fills NaN=0
#R = csr_matrix((dfn['rating'], (dfn['userId'], dfn['movieId']))) 

#if all is well, the shape of the user-item matrix R will be (610, 1235)


#the movies.csv file should also the transformed accordingly

# filter out unpopular movies
#movies = movies[movies['movieId'].isin(movie_ids)]
# redefine movie ids
#movies['movieId'] = movies['movieId'].map(movie_id_map)

def recommend_nn(query, model, Rt, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
        
    # 1. construct new_user-item dataframe given the query
    new_user_dataframe =  pd.DataFrame(query, columns=movies['title'], index=['new_user'])
    # 1.2. fill the NaN
    new_user_dataframe_imputed = new_user_dataframe.fillna(0) #better mean
    # 2. scoring
    # calculates the distances to all other users in the data!
    similarity_scores, neighbor_ids = model.kneighbors(
    new_user_dataframe_imputed,
    n_neighbors=15,
    return_distance=True
    )

    # sklearn returns a list of predictions
    # extract the first and only value of the list

    neighbors_df = pd.DataFrame(
    data = {'neighbor_id': neighbor_ids[0], 'similarity_score': similarity_scores[0]}
    )
    
    # 3. ranking
    # only look at ratings for users that are similar!
    neighborhood = Rt.iloc[neighbor_ids[0]]
  
    
        # filter out movies already seen by the user
    neighborhood_filtered = neighborhood.drop(query.keys(),axis=1)
   

    # calculate the summed up rating for each movie
    # summing up introduces a bias for popular movies
    # averaging introduces bias for movies only seen by few users in the neighboorhood

    df_score = neighborhood_filtered.sum().sort_values(ascending=False)
    
    # return the top-k highest rated movie ids or titles
    df_score_ranked = df_score.sort_values(ascending=False).index.tolist()
    recommended = df_score_ranked[:k]
    return recommended#, df_score

def recommend_nmf(query, model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    
    # 1. construct new_user-item dataframe given the query(votings of the new user)
   
    new_user_dataframe = pd.DataFrame(query, columns=movies['title'], index=['new_user'])
   
    new_user_dataframe_imputed =new_user_dataframe.fillna(0)

    P_new_user_matrix = model.transform(new_user_dataframe_imputed)
    # get as dataframe for a better visualizarion
    P_new_user = pd.DataFrame(P_new_user_matrix, 
                         columns = model.get_feature_names_out(),
                         index = ['new_user'])
    Q_matrix = model.components_
    Q = pd.DataFrame(Q_matrix, columns=movies['title'], index=model.get_feature_names_out())
    R_hat_new_user_matrix = np.dot(P_new_user,Q)
    # get as dataframe for a better visualizarion
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                         columns=movies['title'],
                         index = ['new_user'])
    R_hat_new_filtered = R_hat_new_user#.drop(new_user_query.keys(), axis=1)
    R_hat_new_filtered.T.sort_values(by=["new_user"], ascending=False).index.tolist()
    ranked =  R_hat_new_filtered.T.sort_values(by=["new_user"], ascending=False).index.tolist()
    recommended = ranked[:k]
    return recommended#, R_hat_new_filtered.T.sort_values(by=["new_user"], ascending=False)


BEST_MOVIES = pd.read_csv("best_movies.csv")
BEST_MOVIES.rename(
    index=lambda x: x+1,
    inplace=True
    )
TITLES = ["---"] + list(BEST_MOVIES['title'].sort_values()) 

with open('modelnn.pkl', 'rb') as file:
    DISTANCE_MODEL = pickle.load(file)

with open('nmf_mode.pkl', 'rb') as file:
    NMF_MODEL = pickle.load(file)

# sidebar
with st.sidebar:
    # title
    st.title("It's movie time!")
    # image
    st.image('movie_time.jpg')
    # blank space
    st.write("")
    # selectbox
    page = st.selectbox(
        "what would you like?",
        [
            "--------",
            "popular movies",
            "rate some movies",
            "recommended movies"
            ]
        ) 

if page == "--------":
    # slogan
    st.write("""
    *Movies are like magic tricks (Jeff Bridges)*
    """)
    # blank space
    st.write("")
    # image
    st.image('movie_pics.png')

##########################################################
# Popular Movies
##########################################################

elif page == "popular movies":
    # title
    st.title("Popular Movies")
    col1,col2,col3,col4 = st.columns([10,2,5,5])
    with col1:
        n = st.slider(
        label="how many movies?",
        min_value=1,
        max_value=10,
        value=5
        ) 
    with col3:
        st.markdown("####")
        genre = st.checkbox("include genres")
    with col4:
        st.markdown("###")
        show_button = st.button(label="show movies") 
    
    if genre:
        popular_movies = BEST_MOVIES[['title','genres']]
    else:
        popular_movies = BEST_MOVIES[['title']]
       

    st.markdown("###")
    if show_button:
        st.write(
            HTML(popular_movies.head(n).to_html(escape=False))
            )

##########################################################
# Rate Movies
##########################################################

elif page == "rate some movies":
    # title
    st.title("Rate Movies")
    #
    col1,col2,col3 = st.columns([10,1,5])
    with col1:
        m1 = st.selectbox("movie 1", TITLES)
        st.write("")
        m2 = st.selectbox("movie 2", TITLES)
        st.write("")
        m3 = st.selectbox("movie 3", TITLES)
        st.write("")
        m4 = st.selectbox("movie 4", TITLES)
        st.write("")
        m5 = st.selectbox("movie 5", TITLES) 
    
    with col3:
        r1 = st.slider(
            label="rating 1",
            min_value=1,
            max_value=5,
            value=3
            ) 
        r2 = st.slider(
            label="rating 2",
            min_value=1,
            max_value=5,
            value=3
            ) 
        r3 = st.slider(
            label="rating 3",
            min_value=1,
            max_value=5,
            value=3
            ) 
        r4 = st.slider(
            label="rating 4",
            min_value=1,
            max_value=5,
            value=3
            ) 
        r5 = st.slider(
            label="rating 5",
            min_value=1,
            max_value=5,
            value=3
            ) 

    query_movies = [m1,m2,m3,m4,m5]
    query_ratings = [r1,r2,r3,r4,r5]
    
    user_query = dict(zip(query_movies,query_ratings))

    # get user query
    st.markdown("###")
    user_query_button = st.button(label="save user query") 
    if user_query_button:
        json.dump(
            user_query,
            open("user_query.json",'w')
            )
        st.write("")
        st.write("user query saved successfully")



##########################################################
# Movie Recommendations
##########################################################
else:
    # title
    st.title("Movie Recommendations")
    col1,col2,col3,col4,col5 = st.columns([1,5,1,5,1])
    with col2:
        recommender = st.radio(
            "recommender type",
            ["NMF Recommender","Distance Recommender"]
            )
        st.write("This is under Recomemnder Matthias!")
        


    with col4:
        st.write("###")
        recommend_button = st.button(label="recommed movies")
    

    #load user query
    user_query = json.load(open("user_query.json"))



    # 2. scoring
    
        # calculate the score with the NMF model
    
    
    # 3. ranking
    
        # filter out movies already seen by the user
    
        # return the top-k highest rated movie ids or titles
    
    

    

    if recommend_button:
        if recommender == "NMF Recommender":
            recommend_nmf(user_query, NMF_MODEL, k=10)
            #AgGrid(BEST_MOVIES.head(10))
        elif recommender == "Distance Recommender":
            #AgGrid(BEST_MOVIES.tail(5))
            recommend_nn(user_query, DISTANCE_MODEL, Rt, k=10)
        else:
            st.write("error with chosing recomender system")

    else:
        st.write("Push the button!")

st.write("this is the end! Matze")
#AgGrid(BEST_MOVIES.head(20))
    