# -*- coding: utf-8 -*-

import logging
import pickle
from gensim.models import word2vec

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    file = open("corpus/processed_correct.pkl", "rb")
    sentences = pickle.load(file)
    print(f"Total number of sentences: {len(sentences)}")
    # sentences = word2vec.PathLineSentences("corpus")
    model = word2vec.Word2Vec(
        sentences,
        vector_size=256,
        epochs=32,
    )

    #保存模型，供日後使用
    model.save("word2vec.model")

    #模型讀取方式
    model = word2vec.Word2Vec.load("your_model_name")

if __name__ == '__main__':
    main()