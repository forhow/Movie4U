'''
    TF-IDF

 - 문장간 유사도 점수측정, 추천시스템 구축
'''

import pandas as pd
# vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# mm ; matrix market(TF-IDF score table) 저장 / 로드
from scipy.io import mmwrite, mmread
import pickle


# data load
df_review_1stcs = pd.read_csv('./crawling/one_sentences_review_2018~2021.csv', index_col=0)
print(df_review_1stcs.info())
print(df_review_1stcs.head())


# TF-IDF
tfidf = TfidfVectorizer(sublinear_tf=True)


# TF-IDF matrix 생성
tfidf_matrix = tfidf.fit_transform(df_review_1stcs['reviews'])

# 차후 데이터 추가로 인한 matrix update를 위해 tfidf vectorizing 정보를 저장해둬야 함
with open('./models/tfidf.pickle', 'wb') as f:
    pickle.dump(tfidf, f)

# TF-IDF matrix 저장
mmwrite('./models/tfidf_movie_review.mtx', tfidf_matrix)


