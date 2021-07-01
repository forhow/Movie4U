Project : Movie4U

Introduction : 네이버 영화 리뷰 기반 영화추천

Team : Lightning Crwaling (김광휘, 이도훈, 이가은, 고영익)

Process
 1. Crawling
  - Directory : Movie4U/PRJ02_movie4you/crawling/
  - File(.py) : naver_review_crawling.py
  - Description : 네이버 영화 웹사이트에서 연도별(2018~2021) 영화 제목과 리뷰를 crawling
 
 2. Pre-processing
  - Directory : Movie4U/PRJ02_movie4you
  - 1st step File : pre_preocessing.py
  - 2nd step File : preprocess_one_sentence.py
  - Description 
    > 1st step : alphabet/숫자/특수문자 등 제거, 형태소 분리(동사/명사/형용사 추출), 
                불용어 제거, 분리된 문장 결합, 1차 파일 저장
    > 2nd step : 
