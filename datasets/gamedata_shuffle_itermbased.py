# '''
# Created on Dec 1, 2015

# @author: yin.zheng
# '''

# import numpy as np
# import scipy.io as sio
# import os
# import h5py
# # import cv2
# from scipy.sparse import csr_matrix, lil_matrix
# import random

# # def read_ratings(filename):
# #     ratings = []
# #     with open(filename) as fp:
# #         for line in fp:
# #             user_id, mov_id, rating, time = line.split('::')
# #             ratings.append([int(user_id), int(mov_id), int(rating), int(time)])
# #     return ratings

# # def read_users(filename):
# #     users = []
# #     with open(filename) as fp:
# #         for line in fp:
# #             UserID, Gender, Age, Occupation, Zip_code = line.split('::')
# #             users.append([int(UserID), Gender, int(Age), int(Occupation), Zip_code])
# #     return users

# # def read_movies(filename):
# #     movies = []
# #     with open(filename) as fp:
# #         for line in fp:
# #             MovieID, Title, Genres = line.split('::')
# #             movies.append([int(MovieID), Title, Genres])
# #     return movies
# import pandas as pd
# def read_ratings(filename):
#     ratings = []
#     df = pd.read_csv(filename, sep = ",")
#     # user_id, app_id, is_recommended
#     ratings = df[["user_id", "app_id", "is_recommended", "hours"]].values
#     # with open(filename) as fp:
#     #     for line in fp:
#     #         user_id, mov_id, rating, time = line.split('::')
#     #         ratings.append([int(user_id), int(mov_id), int(rating), int(time)])
#     return ratings

# def read_users(filename):
#     users = []
#     # with open(filename) as fp:
#     #     for line in fp:
#     #         UserID, Gender, Age, Occupation, Zip_code = line.split('::')
#     #         users.append([int(UserID), Gender, int(Age), int(Occupation), Zip_code])
#     df = pd.read_csv(filename, sep = ",")
#     df["Gender"] = 1
#     df["Age"] = 1
#     users = df[["user_id", "Gender", "products", "reviews", "Age"]].values
#     return users

# def read_movies(filename):
#     movies = []
#     # with open(filename) as fp:
#     #     for line in fp:
#     #         MovieID, Title, Genres = line.split('::')
#     #         movies.append([int(MovieID), Title, Genres])
#     df = pd.read_csv(filename, sep = ",")
#     movies = df[["app_id", "title", "tags"]].values
#     return movies


# def write_movie_data(ratings, data_path, output, seed):
    
#     users = {}
#     movs = {}
#     cnt_u = 0
#     cnt_i = 0
#     for user_id, mov_id, rating, _ in ratings:
#         if user_id not in users.keys():
#             users[user_id] = cnt_u
#             cnt_u += 1
#         if mov_id not in movs.keys():
#             movs[mov_id] = cnt_i
#             cnt_i += 1
#     n_users = len(users)
#     n_movies = len(movs)
#     train_ratio = 0.9*0.995
#     valid_ratio = 0.9*0.005
#     test_ratio = 0.1
#     n_ratings = len(ratings)
#     n_test = np.ceil(n_ratings*test_ratio)
#     n_valid = np.ceil(n_ratings*valid_ratio)
#     n_train = n_ratings - n_test - n_valid
    
#     train_input_ratings = np.zeros((n_movies, n_users), dtype='int8')
#     train_output_ratings = np.zeros((n_movies, n_users), dtype='int8')
#     train_input_masks = np.zeros((n_movies, n_users), dtype='int8')
#     train_output_masks = np.zeros((n_movies, n_users), dtype='int8')
    
#     valid_input_ratings = np.zeros((n_movies, n_users), dtype='int8')
#     valid_output_ratings = np.zeros((n_movies, n_users), dtype='int8')
#     valid_input_masks = np.zeros((n_movies, n_users), dtype='int8')
#     valid_output_masks = np.zeros((n_movies, n_users), dtype='int8')
    
#     test_input_ratings = np.zeros((n_movies, n_users), dtype='int8')
#     test_output_ratings = np.zeros((n_movies, n_users), dtype='int8')
#     test_input_masks = np.zeros((n_movies, n_users), dtype='int8')
#     test_output_masks = np.zeros((n_movies, n_users), dtype='int8')
    
    
#     random.seed(seed)
#     random.shuffle(ratings)
#     total_n_train = 0
#     total_n_valid = 0
#     total_n_test = 0
#     cnt = 0
#     for user_id, mov_id, rating, _ in ratings:
#         if cnt < n_train:
#             train_input_ratings[movs[mov_id], users[user_id]] = rating
#             train_input_masks[movs[mov_id], users[user_id]] = 1
#             valid_input_ratings[movs[mov_id], users[user_id]] = rating
#             valid_input_masks[movs[mov_id], users[user_id]] = 1
#             total_n_train += 1
#         elif cnt < n_train+n_valid:
#             valid_output_ratings[movs[mov_id], users[user_id]] = rating
#             valid_output_masks[movs[mov_id], users[user_id]] = 1
#             total_n_valid += 1
#         else:
#             test_output_ratings[movs[mov_id], users[user_id]] = rating
#             test_output_masks[movs[mov_id], users[user_id]] = 1
#             total_n_test += 1
#         cnt += 1
#     test_input_ratings = train_input_ratings + valid_output_ratings
#     test_input_masks = train_input_masks + valid_output_masks        
    
# #     rating_mat = csr_matrix(rating_mat)
    
#     input_r = np.vstack((train_input_ratings, valid_input_ratings, test_input_ratings))
#     input_m = np.vstack((train_input_masks, valid_input_masks, test_input_masks))
#     output_r = np.vstack((train_output_ratings, valid_output_ratings, test_output_ratings))
#     output_m = np.vstack((train_output_masks, valid_output_masks, test_output_masks))
    
#     import os
#     os.makedirs(output, exist_ok=True)

#     f = h5py.File(os.path.join(output, 'gamedata.hdf5'), 'w')
#     input_ratings = f.create_dataset('input_ratings', shape=(n_movies*3, n_users), dtype='int8', data=input_r)
#     input_ratings.dims[0].label = 'batch'
#     input_ratings.dims[1].label = 'movies'
#     input_masks = f.create_dataset('input_masks', shape=(n_movies*3, n_users), dtype='int8', data=input_m)
#     input_masks.dims[0].label = 'batch'
#     input_masks.dims[1].label = 'movies'
#     output_ratings = f.create_dataset('output_ratings', shape=(n_movies*3, n_users), dtype='int8', data=output_r)
#     output_ratings.dims[0].label = 'batch'
#     output_ratings.dims[1].label = 'movies'
#     output_masks = f.create_dataset('output_masks', shape=(n_movies*3, n_users), dtype='int8', data=output_m)
#     output_masks.dims[0].label = 'batch'
#     output_masks.dims[1].label = 'movies'
    
#     split_array = np.empty(
#                            12,
#                            dtype=([
#                                    ('split', 'a', 5),
#                                    ('source', 'a', 14),
#                                    ('start', np.int64, 1),
#                                    ('stop', np.int64, 1),
#                                    ('indices', h5py.special_dtype(ref=h5py.Reference)),
#                                    ('available', np.bool, 1),
#                                    ('comment', 'a', 1)
#                                    ]
#                                   )
#                            )
#     split_array[0:4]['split'] = 'train'.encode('utf8')
#     split_array[4:8]['split'] = 'valid'.encode('utf8')
#     split_array[8:12]['split'] = 'test'.encode('utf8')
#     split_array[0:12:4]['source'] = 'input_ratings'.encode('utf8')
#     split_array[1:12:4]['source'] = 'input_masks'.encode('utf8')
#     split_array[2:12:4]['source'] = 'output_ratings'.encode('utf8')
#     split_array[3:12:4]['source'] = 'output_masks'.encode('utf8')
#     split_array[0:4]['start'] = 0
#     split_array[0:4]['stop'] = n_movies
#     split_array[4:8]['start'] = n_movies
#     split_array[4:8]['stop'] = n_movies*2
#     split_array[8:12]['start'] = n_movies*2
#     split_array[8:12]['stop'] = n_movies*3
#     split_array[:]['indices'] = h5py.Reference()
#     split_array[:]['available'] = True
#     split_array[:]['comment'] = '.'.encode('utf8')
#     f.attrs['split'] = split_array
#     f.flush()
#     f.close()
    
#     f = open(os.path.join(output, 'metadata'), 'w')
#     line = 'n_users:%d\n'%n_users
#     f.write(line)
#     line = 'n_movies:%d'%n_movies
#     f.write(line)
#     f.close()
    
#     f = open(os.path.join(output, 'user_dict'), 'wb')
#     import pickle
#     pickle.dump(users, f)
#     f.close()
    
#     f = open(os.path.join(output, 'movie_dict'), 'wb')
#     pickle.dump(movs, f)
#     f.close()
    
    


# def main(data_path, output, seed):
    
#     ratings = read_ratings(os.path.join(data_path, 'ratings.csv'))
# #     movies = read_movies(os.path.join(data_path, 'movies.dat'))
# #     users = read_users(os.path.join(data_path, 'users.dat'))
    
#     write_movie_data(ratings, data_path, output, seed)

# if __name__ == "__main__":
# #     main("/Users/yin.zheng/Downloads/ml-1m",
# #          "/Users/yin.zheng/ml_datasets/MovieLens1M-shuffle-itembased-0",
# #          1234)
#     base_path = "D:\Empty\CF-NADE-origin\gamedata"
#     print('1')
#     main(base_path,
#          r"gamedata-shuffle-itembased-1",
#          2341)
#     # print( '2')
#     # main(base_path,
#     #      r"gamedata-shuffle-itembased-2",
#     #      3412)
#     # print( '3')
#     # main("/Users/yin.zheng/Downloads/ml-1m",
#     #      "/Users/yin.zheng/ml_datasets/MovieLens1M-shuffle-itembased-3",
#     #      4123)
#     # print( '4')
#     # main("/Users/yin.zheng/Downloads/ml-1m",
#     #      "/Users/yin.zheng/ml_datasets/MovieLens1M-shuffle-itembased-4",
#     #      1324)
# #     from fuel.datasets import H5PYDataset
# #     
# #     trainset = H5PYDataset(os.path.join('/Users/yin.zheng/ml_datasets/MovieLens1M-shuffle-itembased', 'movielens-1m.hdf5'),
# #                            which_sets = ('train',),
# #                            load_in_memory=True,
# #                            sources=('input_ratings', 'output_ratings', 'input_masks', 'output_masks')
# #                            )
# #     print trainset.num_examples
# #     from fuel.schemes import (SequentialScheme, ShuffledScheme,SequentialExampleScheme,ShuffledExampleScheme)
# #     state = trainset.open()
# #     scheme = ShuffledScheme(examples=trainset.num_examples, batch_size=3)
# #     from fuel.streams import DataStream
# #     data_stream = DataStream(dataset=trainset, iteration_scheme=scheme)
# #     for data in data_stream.get_epoch_iterator():
# #         print data[0].shape
    
    
    
import numpy as np
import pandas as pd
import h5py
import os
import random
from typing import List, Dict, Tuple

def read_data(games_path: str, users_path: str, ratings_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the CSV files and return dataframes"""
    games_df = pd.read_csv(games_path)
    users_df = pd.read_csv(users_path)
    ratings_df = pd.read_csv(ratings_path)
    return games_df, users_df, ratings_df

def create_mappings(games_df: pd.DataFrame, users_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Create user and game ID mappings to consecutive integers"""
    users = {user_id: idx for idx, user_id in enumerate(users_df['user_id'].unique())}
    games = {game_id: idx for idx, game_id in enumerate(games_df['app_id'].unique())}
    return users, games

def write_game_data(ratings_df: pd.DataFrame, output_path: str, seed: int,
                    users_dict: Dict, games_dict: Dict) -> None:
    """Write the game data to HDF5 format with train/valid/test splits"""
    
    n_users = len(users_dict)
    n_games = len(games_dict)
    
    # Split ratios
    train_ratio = 0.9 * 0.995
    valid_ratio = 0.9 * 0.005
    test_ratio = 0.1
    
    n_ratings = len(ratings_df)
    n_test = int(np.ceil(n_ratings * test_ratio))
    n_valid = int(np.ceil(n_ratings * valid_ratio))
    n_train = n_ratings - n_test - n_valid
    
    # Initialize rating matrices
    train_input_ratings = np.zeros((n_games, n_users), dtype='int8')
    train_output_ratings = np.zeros((n_games, n_users), dtype='int8')
    train_input_masks = np.zeros((n_games, n_users), dtype='int8')
    train_output_masks = np.zeros((n_games, n_users), dtype='int8')
    
    valid_input_ratings = np.zeros((n_games, n_users), dtype='int8')
    valid_output_ratings = np.zeros((n_games, n_users), dtype='int8')
    valid_input_masks = np.zeros((n_games, n_users), dtype='int8')
    valid_output_masks = np.zeros((n_games, n_users), dtype='int8')
    
    test_input_ratings = np.zeros((n_games, n_users), dtype='int8')
    test_output_ratings = np.zeros((n_games, n_users), dtype='int8')
    test_input_masks = np.zeros((n_games, n_users), dtype='int8')
    test_output_masks = np.zeros((n_games, n_users), dtype='int8')
    
    # Shuffle ratings
    # print(ratings_df.info(), set(ratings_df["is_recommended"].values))
    # return
    ratings_list = ratings_df.values.tolist()
    random.seed(seed)
    random.shuffle(ratings_list)
    
    # Fill matrices
    cnt = 0
    for app_id,helpful,is_recommended,hours,user_id in ratings_list:
        game_id = app_id
        rating = is_recommended + 1
        # print(rating)
        # return
        user_idx = users_dict[user_id]
        game_idx = games_dict[game_id]
        
        if cnt < n_train:
            train_input_ratings[game_idx, user_idx] = rating
            train_input_masks[game_idx, user_idx] = 1
            valid_input_ratings[game_idx, user_idx] = rating
            valid_input_masks[game_idx, user_idx] = 1
        elif cnt < n_train + n_valid:
            valid_output_ratings[game_idx, user_idx] = rating
            valid_output_masks[game_idx, user_idx] = 1
        else:
            test_output_ratings[game_idx, user_idx] = rating
            test_output_masks[game_idx, user_idx] = 1
        cnt += 1
    
    test_input_ratings = train_input_ratings + valid_output_ratings
    test_input_masks = train_input_masks + valid_output_masks
    
    # Stack all matrices
    input_r = np.vstack((train_input_ratings, valid_input_ratings, test_input_ratings))
    input_m = np.vstack((train_input_masks, valid_input_masks, test_input_masks))
    output_r = np.vstack((train_output_ratings, valid_output_ratings, test_output_ratings))
    output_m = np.vstack((train_output_masks, valid_output_masks, test_output_masks))
    
    print(f"Input Ratings Shape: {input_r.shape}")
    print(f"Output Ratings Shape: {output_r.shape}")
    print(f"Input Masks Shape: {input_m.shape}")
    print(f"Output Masks Shape: {output_m.shape}")
    print(f"First few entries of input_r: {input_r[:5]}")
    print(f"First few entries of output_r: {output_r[:5]}")
    print(f"First few entries of input_m: {input_m[:5]}")
    print(f"First few entries of output_m: {output_m[:5]}")
    # return

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Write HDF5 file
    with h5py.File(os.path.join(output_path, 'game-ratings.hdf5'), 'w') as f:
        # Create datasets
        input_ratings = f.create_dataset('input_ratings', shape=(n_games*3, n_users), dtype='int8', data=input_r)
        input_ratings.dims[0].label = 'batch'
        input_ratings.dims[1].label = 'games'
        
        input_masks = f.create_dataset('input_masks', shape=(n_games*3, n_users), dtype='int8', data=input_m)
        input_masks.dims[0].label = 'batch'
        input_masks.dims[1].label = 'games'
        
        output_ratings = f.create_dataset('output_ratings', shape=(n_games*3, n_users), dtype='int8', data=output_r)
        output_ratings.dims[0].label = 'batch'
        output_ratings.dims[1].label = 'games'
        
        output_masks = f.create_dataset('output_masks', shape=(n_games*3, n_users), dtype='int8', data=output_m)
        output_masks.dims[0].label = 'batch'
        output_masks.dims[1].label = 'games'
        
        # Create split attributes
        split_array = np.empty(
            12,
            dtype=([
                ('split', 'a', 5),
                ('source', 'a', 14),
                ('start', np.int64, 1),
                ('stop', np.int64, 1),
                ('indices', h5py.special_dtype(ref=h5py.Reference)),
                ('available', np.bool_, 1),
                ('comment', 'a', 1)
            ])
        )
        
        split_array[0:4]['split'] = 'train'.encode('utf8')
        split_array[4:8]['split'] = 'valid'.encode('utf8')
        split_array[8:12]['split'] = 'test'.encode('utf8')

        split_array[0:12:4]['source'] = 'input_ratings'.encode('utf8')
        split_array[1:12:4]['source'] = 'input_masks'.encode('utf8')
        split_array[2:12:4]['source'] = 'output_ratings'.encode('utf8')
        split_array[3:12:4]['source'] = 'output_masks'.encode('utf8')
        
        split_array[0:4]['start'] = 0
        split_array[0:4]['stop'] = n_games
        split_array[4:8]['start'] = n_games
        split_array[4:8]['stop'] = n_games*2
        split_array[8:12]['start'] = n_games*2
        split_array[8:12]['stop'] = n_games*3
        
        split_array[:]['indices'] = h5py.Reference()
        split_array[:]['available'] = True
        split_array[:]['comment'] = '.'.encode('utf8')
        
        f.attrs['split'] = split_array
    
    # Write metadata
    with open(os.path.join(output_path, 'metadata'), 'w') as f:
        f.write(f'n_users:{n_users}\n')
        f.write(f'n_games:{n_games}')
    
    # Write dictionaries
    import pickle
    with open(os.path.join(output_path, 'user_dict'), 'wb') as f:
        pickle.dump(users_dict, f)

    with open(os.path.join(output_path, 'game_dict'), 'wb') as f:
        pickle.dump(games_dict, f)

def main(games_path: str, users_path: str, ratings_path: str, output_path: str, seed: int) -> None:
    """Main function to process game data and create HDF5 files"""
    # Read data
    games_df, users_df, ratings_df = read_data(games_path, users_path, ratings_path)
    
    # Create ID mappings
    users_dict, games_dict = create_mappings(games_df, users_df)
    
    # Write data to HDF5
    write_game_data(ratings_df, output_path, seed, users_dict, games_dict)

if __name__ == "__main__":
    # Example usage
    base_path = "D:\Empty\CF-NADE-origin\gamedata"
    main(
        games_path=r"D:\Empty\CF-NADE-origin\gamedata\games.csv",
        users_path=r"D:\Empty\CF-NADE-origin\gamedata\users.csv",
        ratings_path=r"D:\Empty\CF-NADE-origin\gamedata\ratings.csv",
        output_path="gamedata-shuffle-itembased-1",
        seed=2341
    )