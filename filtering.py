# %%
import pandas as pd

films = pd.read_csv('ml-latest-small/movies.csv')

print(films.head())

# %%
genres = films['genres'].str.split('|')

genres

# %%
movies = films.copy()
movies.loc[:, 'genres'] = genres

print(movies.head())

# %% [markdown]
# 

# %%
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

df = movies.copy()

df_binarized = mlb.fit_transform(df['genres'])

df_binarized

# %%
list(mlb.classes_)

# %%
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

cosine_similarities = 1-pairwise_distances(df_binarized, metric='cosine')

print(cosine_similarities[:5, :5])

# %%
movie = df[df['title'].str.contains('age of ultron', case=False, na=False)].index[0]
movie

cosine_similarities[2235, 9219]

# %%
def movie_recommender(movie_name):
    df.reset_index()
    movie = df[df['title'].str.contains(movie_name.title(), case=False, na=False)].index[0]
    similar_movies = cosine_similarities[movie].argsort()[::-1][1:6]
    recommendation = df.iloc[similar_movies]['title']
    return recommendation

movie_recommender(input())

# %%
ratings = pd.read_csv('ml-latest-small/ratings.csv')
df = ratings.merge(movies, on='movieId')
df = df.drop(columns=['timestamp'])

print(df.head())

# %%
from torchvision.transforms.functional import to_tensor

useritem_matrix = df.pivot_table(index='userId', columns='title', values='rating', fill_value=0)
useritem_matrix = useritem_matrix.to_numpy()
useritem_tensor = to_tensor(useritem_matrix)
useritem_tensor.shape

# %%
from torch.linalg import svd

useritem_matrix_svd = svd(useritem_tensor, full_matrices=True)
useritem_matrix_svd

# %%
import torch

svd_U = useritem_matrix_svd[0]
svd_S = useritem_matrix_svd[1]
svd_Vh = useritem_matrix_svd[2]

# %%
print(svd_U.shape, svd_S.shape, svd_Vh.shape)


# %%
svd_U_mat = svd_U
svd_S_mat = torch.diag_embed(svd_S)
svd_Vh_mat = svd_Vh[:, :610, :]

svd_matrix = svd_U_mat @ svd_S_mat @ svd_Vh_mat
svd_matrix

# %%
torch.allclose(useritem_tensor, svd_matrix, atol=1e-6)  # Adjust tolerance if needed

# %%
from torch import topk

energy = torch.cumsum(svd_S,  dim=-1)/torch.sum(svd_S, dim=-1, keepdim=True)
k = (energy >= 0.95).nonzero(as_tuple=True)[1][0].item()

U_tk = svd_U[:, :, :k]
S_tk = torch.diag_embed(svd_S[:, :k])
Vh_tk = svd_Vh[:, :k, :]

svd_matrix_tk = U_tk @ S_tk @ Vh_tk
svd_matrix_tk

# %%
mask = useritem_tensor != 0
l1_error = torch.abs(useritem_tensor[mask] - svd_matrix_tk[mask]).mean()
l2_error = torch.sqrt(torch.mean((useritem_tensor[mask] - svd_matrix_tk[mask]) ** 2))
l1_error, l2_error

# %%
print((useritem_tensor == 0).sum())

# %%
filled_ui_tensor = torch.where(useritem_tensor == 0, svd_matrix_tk, useritem_tensor)
filled_ui_tensor

# %%
print((filled_ui_tensor == 0).sum()) 


# %%



