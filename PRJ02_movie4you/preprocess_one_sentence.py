'''
    영화 하나당 하나의 리뷰 문장으로 병합
'''

import pandas as pd

print('전처리 수행할 파일 이름과 년도 확인')
print('ex) cleaned_review_2021.csv-> 2021')
year = int(input('대상 파일의 연도 입력 : ' ))

# csv 파일 로드
df = pd.read_csv('./crawling/cleaned_review_{}.csv'.format(year), index_col=0)
df.dropna(inplace=True)

# 병합된 리뷰가 저장될 리스트 준비
one_sentences = []


# 영화 제목별로 하나의 리뷰를 생성
# print(df['titles'].unique())
for title in df['titles'].unique():
    # 영화 제목별 모든 리뷰 추출
    temp = df[df['titles']==title]['cleaned_sentences']
    # print(title)
    # print(temp)

    # 추출된 모든 리뷰 병합
    one_sentence = ' '.join(temp)

    # 병합된 리뷰 리스트에 내용 추가
    one_sentences.append(one_sentence)

# 영화 제목과 병합된 리뷰로 구성된 데이터 프레임 생성
df_one_sentences = pd.DataFrame({'titles':df['titles'].unique(),
                                 'reviews':one_sentences})
print(df_one_sentences)

# csv 파일로 저장
df_one_sentences.to_csv('./crawling/one_sentences_review_{}.csv'.format(year))