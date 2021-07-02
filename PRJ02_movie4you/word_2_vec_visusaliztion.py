'''
    Word 2 vec 수행 후 단어들의 배치를 시각화 확인

'''

import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import font_manager, rc
import matplotlib as mpl

font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus']=False
rc('font', family=font_name)

embedding_model = Word2Vec.load('./models/word2VecModel_2018_2021.model')
key_word = '번개'
sim_word = embedding_model.wv.most_similar(key_word, topn=10)
print(sim_word)