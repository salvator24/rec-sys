# Author: shivam shakti 
# Date: 2017-12-18 23:47:08 
# Last Modified by:   shivam shakti 
# Last Modified time: 2017-12-18 23:47:08 

import os, sys

src_dir = os.path.join(os.getcwd(), os.pardir, os.pardir, 'src')
data_dir = os.path.join(os.getcwd(), os.pardir, os.pardir, 'data')
sys.path.append(src_dir)

import pandas as pd
import numpy as np
import pickle as pkl
import multiprocessing
from gensim.models import Word2Vec

class Features(object):
    '''class to extract features'''
    
    def __init__(self):
        '''constructor'''
    
    def extract_data(self):
        '''extract movie liked by users as a sequence'''

        # load CSV file
        ratings_df = pd.read_csv(os.path.join(data_dir, 'raw/ml-20m/ratings.csv'))

        # filter ratings equal and above 4.5 to be liked movies
        ratings_df = ratings_df[ratings_df.rating >= 4.5]

        # drop `rating` column 
        ratings_df.drop(['rating'], axis=1, inplace=True)

        # group by userId and sort the movies by their timestamp and create a sequence of movies
        movie_list = list()
        for user_id, data in list(ratings_df.sort_values('timestamp').groupby('userId')):
            movie_list.append(data.movieId.astype(str).tolist())
        
        # save extracted movie_list
        with open(os.path.join(data_dir, 'interim/movie_list.pkl'), 'wb') as f:
            pkl.dump(movie_list, f)

        print (movie_list[:5])

    def build_embeddings(self):
        '''build movie embeddings'''

        # load movie_list data
        with open(os.path.join(data_dir, 'interim/movie_list.pkl'), 'rb') as f:
            movie_list = pkl.load(f)
        
        # create and save movie2vec model
        cores = multiprocessing.cpu_count()-1
        model = Word2Vec(movie_list, min_count=1, size=200, sg=1, iter=2, negative=10, workers=cores)
        model.save(os.path.join(data_dir, 'processed/movie2vec.model'))

        # load movie2vec model
        movies = model.wv.vocab.keys()
        movie_size = len(movies)

        # index to movie dictionary
        idx_to_movie = dict(enumerate(list(movies)))

        # movie to index dictionary
        movie_to_idx = {v: k for k, v in idx_to_movie.items()}

        # maximum length of movie sequence
        max_length = max([len(i) for i in movie_list])
        
        # add padding to make aequence length same
        for i, _ in enumerate(movie_list):
            movie_list[i] = [movie_to_idx[movie] for movie in movie_list[i]] + [0] * (max_length - len(movie_list[i]))
        
        # convert list to numpy array
        movie_list = np.asarray(movie_list)

        # create embedding matrix
        vec_X = np.zeros([movie_size+1, 200])
        for i, j in idx_to_movie.items():
            vec_X[i] = model[j]
        vec_X[movie_size]=np.ones(200)

        np.savez(os.path.join(data_dir, 'processed/data.npz'), movie_list=movie_list)
        np.savez(os.path.join(data_dir, 'processed/embed.npz'), embed=vec_X)


if __name__ == "__main__":
    feature = Features()
    # feature.extract_data()
    feature.build_embeddings()