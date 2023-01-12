# -*- coding: utf-8 -*-

import logging
import pickle
from gensim.models import word2vec

def main(epoch: int, process_type: str):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if process_type == 'slm':
        file = open("corpus/processed_correct.pkl", "rb")
    elif process_type == 'raw':
        file = open("corpus/combined.pkl", "rb")
    sentences = pickle.load(file)

    print(f"Total number of sentences: {len(sentences)}")
    # sentences = word2vec.PathLineSentences("corpus")
    model = word2vec.Word2Vec(
        sentences,
        sg=0,
        vector_size=256,
        window=5,
        min_count=5,
        epochs=epoch,
        hs=0,
        negative=5,
        sample=1e-4,
        workers=20,
    )

    #保存模型，供日後使用
    save_name = f"cbow_wiki_msr_pku_e{epoch}_{process_type}.model"
    model.save(save_name)

if __name__ == '__main__':
    PROCESS_TYPE = 'slm'
    EPOCH = 50
    main(EPOCH, PROCESS_TYPE)
