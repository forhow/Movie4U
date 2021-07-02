'''
    Word2Vec

    - Vectorizing
    - NLP model embedding 관련
'''

import pandas as pd
from gensim.models import Word2Vec

# data load
review_word = pd.read_csv('./crawling/cleaned_review_2018_2021.csv', index_col=0)
print(review_word.info())
print(review_word.head())

# review part separation
cleaned_token_review = list(review_word['cleaned_reviews'])
print(len(cleaned_token_review))

cleaned_tokens = []
count = 0
# tokenization
for sentence in cleaned_token_review:
    token = sentence.split()
    cleaned_tokens.append(token)
print(len(cleaned_tokens))

# 벡터공간의 근처에 배치된 단어는 유사하다는 전체로 학습
# embedding_model = Word2Vec(cleaned_tokens,
#                            vector_size=100, # 차원축소 output dim
#                            window=4, # 커널 사이즈 (문장의 길이)
#                            min_count=20, # 출현빈도가 20이상인 단어만 사용
#                            workers=4,  # 사용할 cpu core 개수
#                            epochs=100, # 학습 진행 횟수
#                            sg=1) # algorithm 선택
# embedding_model.save('./models/word2VecModel_2018_2021.model')


# tokenization 된 상태 확인  -> deprecated
# print(embedding_model.wv.vocab.keys())
# print(len(embedding_model.wv.vocab.keys()))
'''

'''