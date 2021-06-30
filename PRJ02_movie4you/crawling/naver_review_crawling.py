'네이버 영화 리뷰 크롤링'
import time

'''
Project : 영화 리뷰기반 추천 시스템
- 영화별로 리뷰 크롤링
- 개봉년도별 
- Word2Vec 사용예정
- 모델링은 최소화
- 시각화 부분 
'''

'''
수행작업명 : Crwaling

수행방법 : 
    1. 각자 2019년 내용 크롤링 진행 
    2. 가장 먼저 완성된 코드로 그 외 년도 크롤링 수행
    3. 크롤링 완료 데이터(파일) 종합해서 raw data 생성

작업/완료 데이터 형식 : Pandas - Dataframe
컬럼 : ['years', 'titles', 'reviews']

작업/완료 파일 형식: .csv
파일명 : reviews_0000.csv 

코드/결과물 저장소 : Google Drive
'''
from selenium import webdriver
import pandas as pd
from selenium.common.exceptions import NoSuchElementException
import time

options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox') # 윈도우에선 주지 않아도 되는 옵션, 리눅스에서 사용
options.add_argument('--disable-dev-shm-usage') # 필수, 없으면 에러
options.add_argument('disable-gpu') # 윈도우에선 주지 않아도 되는 옵션, 리눅스에서 사용
options.add_argument('lang=ko_KR') # 언어처리
driver = webdriver.Chrome('./chromedriver.exe', options=options)
years = []
titles = []
reviews = []

try:
    for i in range(1, 3): # 2019년 개봉작 page수
        url = 'https://movie.naver.com/movie/sdb/browsing/bmovie.nhn?open={}&page={}'.format('2019', i)
        time.sleep(0.5)
        for j in range(1, 3):#len(y)+1): # page당 영화 목록
            try:
                driver.get(url)
                time.sleep(0.5)
                movie_title_xpath = '//*[@id="old_content"]/ul/li[{}]/a'.format(j)
                title = driver.find_element_by_xpath(movie_title_xpath).text
                driver.find_element_by_xpath(movie_title_xpath).click()
                print(title)
                try:
                    btn_review_xpath = '//*[@id="movieEndTabMenu"]/li[6]/a/em'
                    driver.find_element_by_xpath(btn_review_xpath).click()  # 리뷰 버튼 클릭
                    time.sleep(0.5)
                    review_len_xpath = '//*[@id="reviewTab"]/div/div/div[2]/span/em'
                    review_len = driver.find_element_by_xpath(review_len_xpath).text
                    review_len = int(review_len)
                    try:
                        for k in range(1, 3):#((review_len-1) // 10) + 2):
                            review_page_xpath = '//*[@id="pagerTagAnchor{}"]/span'.format(k)
                            driver.find_element_by_xpath(review_page_xpath).click()
                            time.sleep(0.5)

                            for l in range(1, 11):
                                review_title_xpath = '//*[@id="reviewTab"]/div/div/ul/li[{}]'.format(l)
                                try:
                                    driver.find_element_by_xpath(review_title_xpath).click()
                                    time.sleep(0.5)
                                    try:
                                        review_xpath = '//*[@id="content"]/div[1]/div[4]/div[1]/div[4]'
                                        review = driver.find_element_by_xpath(review_xpath).text
                                        titles.append(title)
                                        reviews.append(review)
                                        driver.back()
                                        time.sleep(0.5)
                                    except:
                                        driver.back()
                                        time.sleep(0.5)
                                        print('review crawling error')
                                except:
                                    print('review error')
                    except:
                        print('review page btn click err')
                except NoSuchElementException:
                    driver.get(url)
                    time.sleep(0.5)
                    print('review btn is not find')
            except NoSuchElementException:
                print('NoSuchElementException 입니다.')
    df_review = pd.DataFrame({'titles':titles, 'reviews':reviews})
    df_review['years'] = 2019
    print(df_review.head(20))
    df_review.to_csv('./reviews_2019.csv')

except:
    print('except1')
finally:
    driver.close()