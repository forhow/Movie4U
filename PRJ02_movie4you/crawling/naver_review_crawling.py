'네이버 영화 리뷰 크롤링'

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
컬럼 : ['years','titles', 'reviews']

작업/완료 파일 형식: .csv
파일명 : reviews_0000.csv 

코드/결과물 저장소 : Google Drive
'''

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import requests
import time


# 'TEST'
# url ='https://movie.naver.com/movie/sdb/browsing/bmovie.nhn?open=2019&page=1'
# driver.get(url)
# driver.find_element_by_xpath('//*[@id="old_content"]/ul/li[1]/a').click()
# time.sleep(0.5)
# driver.find_element_by_xpath('//*[@id="movieEndTabMenu"]/li[6]/a/em').click()
# driver.find_element_by_xpath('//*[@id="reviewTab"]/div/div/ul/li[1]').click()


# 2019 개봉영화 page 1~44

# Browser Options
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('disable_gpu')
options.add_argument('lang=ko_KR')

driver = webdriver.Chrome('chromedriver', options=options)
titles = []
reviews = []

# Crawling 대상 정보 입력
year = int(input('크롤링 대상 년도 입력 (ex. 2021): '))
page = int(input('해당 연도의 영화개수 입력 (ex. 608): '))
page = (page // 20) +2

# Crawling 수행
try:
    # 년도별 개봉영화 목록 페이지
    for i in range(1, page):
        url = 'https://movie.naver.com/movie/sdb/browsing/bmovie.nhn?open={}&page={}'.format(year, i)

        # 페이지당 영화 목록 수 (20개)
        for j in range(1,21):
            try:
                # 해당 년도 영화목록 페이지 접속
                driver.get(url)
                time.sleep(1)

                # 영화 클릭 (영화 상세페이지 이동)
                movie_title_xpath = '//*[@id="old_content"]/ul/li[{}]/a'.format(j)
                title = driver.find_element_by_xpath(movie_title_xpath).text
                print(title)
                driver.find_element_by_xpath(movie_title_xpath).click()
                time.sleep(1)
                try:
                    # 영화 상세 페이지에서 리뷰버튼 클릭
                    btn_review_xpath = '//*[@id="movieEndTabMenu"]/li[6]/a/em'
                    driver.find_element_by_xpath(btn_review_xpath).click()
                    time.sleep(1)

                    # 리뷰 개수 확인 및 리뷰 페이지 계산
                    review_len_xpath = '//*[@id="reviewTab"]/div/div/div[2]/span/em'
                    review_len = driver.find_element_by_xpath(review_len_xpath).text
                    review_len = int(review_len)

                    try:
                        # 리뷰 페이지 선택
                        for k in range(1, ((review_len-1) // 10)+2):
                            review_page_xpath = '//*[@id="pagerTagAnchor{}"]/span'.format(k)
                            driver.find_element_by_xpath(review_page_xpath).click()
                            time.sleep(1)

                            # 리뷰 선택 및 크롤링 수행
                            for l in range(1,11):
                                review_title_xpath = '//*[@id="reviewTab"]/div/div/ul/li[{}]'.format(l)
                                try:
                                    # 리뷰 선택
                                    driver.find_element_by_xpath(review_title_xpath).click()
                                    time.sleep(1)
                                    try:
                                        # 영화제목 및 리뷰 크롤링
                                        review_xpath = '//*[@id="content"]/div[1]/div[4]/div[1]/div[4]'
                                        review = driver.find_element_by_xpath(review_xpath).text
                                        titles.append(title)
                                        reviews.append(review)
                                        driver.back()
                                        time.sleep(1)
                                    except:
                                        driver.back()
                                        time.sleep(1)
                                        print('review crawling error')
                                except:
                                    time.sleep(1)
                                    print('review title click error')
                    except:
                        print('review page btn click error')
                except:
                    print('review btn click error')

            except NoSuchElementException:
                driver.get(url)
                time.sleep(1)
                print('NoSuchElementException')
        print(len(reviews))

    # 크롤링 결과 Dataframe 생성
    df_review = pd.DataFrame({'titles':titles, 'reviews':reviews})
    df_review['years'] = year
    print(df_review.head(20))

    # CSV 파일 저장
    df_review.to_csv('./reviews_{}.csv'.format(year))


except:
    print('except1')
finally:
    driver.close()



















