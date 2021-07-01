'''
    영화 하나당 하나의 리뷰 문장으로 병합
'''

import pandas as pd

print('전처리 수행할 파일 이름과 년도 확인')
print('ex) cleaned_review_2021.csv-> 2021')
year = int(input('대상 파일의 연도 입력 : ' ))

df = pd.read_csv('./crawling/cleaned_review_{}.csv'.format(year), index_col=0)
df.dropna(inplace=True)
one_sentences = []

# print(df['titles'].unique())

for title in df['titles'].unique():
    temp = df[df['titles']==title]['cleaned_sentences']
    print(title)
    print(temp)
    one_sentence = ' '.join(temp)
    one_sentences.append(one_sentence)

df_one_sentences = pd.DataFrame({'titles':df['titles'].unique(),
                                 'reviews':one_sentences})
print(df_one_sentences)
df_one_sentences.to_csv('./crawling/one_sentences_review_{}.csv'.format(year))