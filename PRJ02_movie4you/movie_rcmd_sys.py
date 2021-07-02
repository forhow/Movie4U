import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
# from gensim.models import
from scipy.io import mmread, mmwrite
import pickle

df_review_1stcs = pd.read_csv('./crawling/one_sentences_review_2018~2021.csv', index_col=0)
# print(df_review_1stcs.info())
# print(df_review_1stcs.head())


' enumerate'
ls = '겨울왕국 라이온킹 알라딘'.split()
print(list(enumerate(ls)))


exit()
# TF-IDF matrix load / TF-IDF load
tfidf_matrix = mmread('./models/tfidf_movie_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    tfidf = pickle.load(f)



def getRcommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x : x[1], reverse=True)
    simScore = simScore[1:10] # 0은 self
    movieidx = [i[0] for i in simScore]
    recMovieList = df_review_1stcs.iloc[movieidx, 0] # row indexing
    return recMovieList



# 영화의 index 탐색
movie_idx = df_review_1stcs[df_review_1stcs['titles']=='기생충 (PARASITE)'].index[0]

# 또는 index를 직접 지정해서 사용 가능
# movie_idx = 300
# 영화 제목 확인
print(df_review_1stcs.iloc[movie_idx, 0])

cosine_sim = linear_kernel(tfidf_matrix[movie_idx], tfidf_matrix)
recommendation = getRcommendation(cosine_sim)
print(recommendation)