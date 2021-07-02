'''
    Word Cloud Visualization

'''

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections
from konlpy.tag import Okt
import matplotlib as mpl
from matplotlib import font_manager, rc

# mpl에 font 정보 제공

'''
# for Windows
    font_path = "C:/Windows/Fonts/Malgun.TTF"  # 사용할 ttf 파일 경로 지정
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
    mpl.font_manager._rebuild()

# for Linux
    import matplotlib.font_manager as fm
    fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
    font = fm.FontProperties(fname=fontpath, size=9)
    plt.rc('font', family='NanumBarunGothic'
    mpl.font_manager._rebuild()
'''
fontpath = 'malgun.ttf'
font_name = font_manager.FontProperties(fname=fontpath).get_name()
rc('font', family=font_name)
mpl.font_manager._rebuild()

df = pd.read_csv('./crawling/cleaned_review_2020.csv', index_col=0)
# df = pd.read_csv('one_sentences_review_2020.csv', index_col=0)

df.dropna(inplace=True)

print(df.info())
print(df.titles.unique())

# 해당하는 조건의 index를 반환
movie_index = df[df['titles']=='기기괴괴 성형수 (Beauty Water)'].index[0]
print(movie_index)

# 리뷰를 구성하는 단어를 확인
print(df.cleaned_sentences[movie_index])
words = df.cleaned_sentences[movie_index].split()
print(words)

# word cloud를 그리기 위해 단어와 해당 단어의 빈도수를 함께 넘겨줘야 함
# Counter : unique 한 값을 추출하며 count를 함께 반환
word_dict = collections.Counter(words)  # dict 형식의 counter 객체로 반환
print(word_dict)
word_dict = dict(word_dict)  # dict로 변환
print(word_dict)


stopwords = ['관객', '작품', '받다', '촬영', '크다', '메다', '리뷰',
             '개봉', '스크린', '출연', '극장', '평가', '출연', '평점']
print(stopwords)


# word cloud 그리기
'''
method_1 : generate
- 구성된 단어를 기준으로 그림 ; 단어 리스트로 전달
- stopword 지정함으로써 추가적인 전처리 과정 수행 가능
'''
word_cloud_img = WordCloud(background_color='white', # 배경색 설정
                           max_words=200,  # 사용할 단어 개수 제한
                           stopwords=stopwords,
                           font_path=fontpath
                           ).generate(df.cleaned_sentences[movie_index])

'''
method_2 : generate_from_frequencies
- 단어 출현 빈도에 따라 그림 ; {단어:count} 형태의 dictionary로 값 전달 
- stopword 추가 제거 불가
'''
# word_cloud_img = WordCloud(background_color='white',
#                            max_words=200,
#                            font_path=fontpath
#                            ).generate_from_frequencies(word_dict)

plt.figure(figsize=(8,8))
plt.imshow(word_cloud_img, interpolation='bilinear') #이미지의 부드럽기 정도설정
plt.title(df.titles[movie_index])
plt.axis('off')
plt.show()