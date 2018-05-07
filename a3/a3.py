# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
# Note that I have not provided many doctests for this one. I strongly
# recommend that you write your own for each function to ensure your
# implementation is correct.

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    token = []
    
    for i in range(len(movies)):
        token.append(tokenize_string(movies.genres[i]))
        #print(token[i])
    
    movies['tokens'] = pd.Series(token, movies.index)
    #print(movies['tokens'])
    
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    count = 0
    tokens = []
    csr_list = []
    vocab = defaultdict(lambda:0)
    term_data = Counter()
    term_doc = Counter()
    tf = 0
    
    for i in movies.tokens:
        tokens.extend(i)
        #print(tokens)
        
    #tokens = sorted(set(tokens))
    
    for token in sorted(set(tokens)):
        vocab[token] = count
        #print(vocab)
        count += 1
    
    for t in movies.tokens:
        term_data.update(t)
        #print(term_data)
    
    #print(term_data)
    
    """
    c = Counter()
    listoi = ['a', 'b', 'a', 'c', 'a', 'b']
    
    for i in listoi:
        c.update(i)
    
    print(c)
    """
    
    for i in range(len(movies)):
        #print(i)
        term_doc.clear()
        term_doc.update(movies.tokens[i])
        #print(term_doc)
        #print(term_doc.values())
        num = max(term_doc.values())
        #print(num)
        sorted_term_doc = sorted(set(movies.tokens[i]))
        #print(sorted_term_doc)
        rows = []
        cols = []
        values = []
        for t in sorted_term_doc:
            rows.append(0)
            cols.append(vocab[t])
            tf = (term_doc[t]/ num*(math.log10(len(movies)/term_data[t])))
            values.append(tf)
            #print(values)
            x = csr_matrix((values,(rows,cols)), shape=(1, len(vocab)))
        csr_list.append(csr_matrix((values, (rows,cols)), shape=(1, len(vocab))))
    
    movies['features'] = pd.Series(csr_list, index=movies.index)
    
    return tuple((movies, vocab))


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    return (a.dot(b.T).toarray()[0][0]) / (np.linalg.norm(a.toarray()) * np.linalg.norm(b.toarray()))


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    #print(ratings_train.userId)
    train_movie_id = defaultdict(lambda:0)
    train_ratings = defaultdict(lambda:0)
    
    test_ratings = []
    
    for i in (ratings_train.index):
        #print(i)
        train_movie_id[i] = ratings_train.movieId[i]
        train_ratings[i] = ratings_train.rating[i]
    
    for j in (ratings_test.index):
        test_user_id = ratings_test.userId[j]
        test_movie_id = ratings_test.movieId[j]
        movie_index = []
        movie_index = ratings_train.movieId[ratings_train.userId==test_user_id].index
        a_csr = movies.loc[movies.movieId==test_movie_id].squeeze()['features']
        flag = True
        weighted_avg_rating = 0
        cosine_similarity = 0
        total_rate = 0
        
        for k in movie_index:
            b_csr = movies.loc[movies.movieId == train_movie_id[k]].squeeze()['features']
            cosine_similarity = cosine_sim(a_csr, b_csr)
            if cosine_similarity > 0:
                weighted_avg_rating += cosine_similarity * train_ratings[k]
                total_rate += cosine_similarity
                flag = False
        if flag == False:
            test_ratings.append(weighted_avg_rating/total_rate)
        else:
            test_ratings.append(ratings_train.rating[ratings_train.userId == test_user_id].mean())
        
    return np.array(test_ratings)


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()