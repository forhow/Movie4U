''' this is for test  ->> for pre-processing'''


'''
    Pre-processing
    
    1. csv file load
    2. alphabet, number, symbol 등 제거
    3. tokenization (형태소 단위로 문장 분리 by okt, kkma, komoran)
    4. 명사, 동사, 형용사 token 추출
    5. 불용어 제거
    6. 분리된 문장 병합 (cleaned_sentences)
    7. 2 columns(title, cleaned_sentences) dataframe 및 csv 파일로 저장

'''
# 형태소 분리기 3종 import
from konlpy.tag import Okt
from konlpy.tag import Komoran
from konlpy.tag import Kkma

import pandas as pd
import numpy as np
import re

print('전처리 수행할 파일 이름과 년도 확인')
print('ex) reviews_2021.csv -> 2021')
year = int(input('대상 파일의 연도 입력 : ' ))


'Setting / Initialization'
# 형태소 분리 객체 생성
okt = Okt()

# progress counter 용
count = 0

# 불용어 제거 후 단어 병합 저장용 리스트
cleaned_sentences = []

# 불용어 리스트(set)
stopwords = pd.read_csv('./crawling/stopwords.csv', index_col=0)
movie_stopwords = '영화 작품 배우 주인공 주연 조연 감독 연출 극본 시나리오'.split()
stopwords_list = list(stopwords.stopword) + movie_stopwords
stopwords_list = set(stopwords_list)
# print(len(stopwords_list))
# print(type(stopwords), len(stopwords), stopwords)



# 1. csv file load
df = pd.read_csv('./crawling/reviews_{}.csv'.format(year), index_col=0 )
print(df.info())
print(df.head())


for sentence in df.reviews:
    # progress counter
    count += 1
    if count % 10 == 0:
        print('.', end='')
    if count % 100 == 0:
        print('')

    # 2. alphabet, number, symbol 등 제거
    sentence = re.sub('[^가-힣 | ' ']', '', sentence)

    # 3. tokenization
    token = okt.pos(sentence, stem=True)
    df_token = pd.DataFrame(token, columns=['word', 'class'])

    # 4. 명사, 동사, 형용사 token 추출
    df_cleaned_token = df_token[(df_token['class'] == 'Noun') |
                            (df_token['class'] == 'Verb') |
                            (df_token['class'] == 'Adjective')]
    words = []

    # 5. 불용어 제거
    for word in df_cleaned_token['word']:
        if len(word) > 1:
            if word not in stopwords_list:
                words.append(word)

    # 6. 분리된 문장 병합 (cleaned_sentences)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)

# 7. dataframe 및 csv 파일로 저장
df['cleaned_sentences'] = cleaned_sentences
# print(df.head())

df = df[['titles', 'cleaned_sentences']]
print(df.info())
df.to_csv('./crawling/cleaned_review_{}.csv'.format(year))
