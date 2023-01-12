from gensim.models import word2vec
import numpy as np

model_name = "cbow_wiki_msr_pku_e50_raw.model"
model = word2vec.Word2Vec.load(model_name)

specials = ["<BOS>", "<UNK>", "<PUNC>", "<NUM>", "的", "</s>", "国", "<ENG>"]
remains = set(model.wv.key_to_index.keys()) - set(specials)
vocab = specials
vocab.extend(list(remains))

with open('cbow/vocab.txt', "w") as f:
    for v in vocab:
        f.write(v)
        f.write("\n")

for v in vocab:
    try:
        if v == "<BOS>":
            word_vectors = np.zeros(256, dtype='float32')
        else:
            word_vectors = np.vstack([word_vectors, model.wv[v]])
    except KeyError:
        zero_vectors = np.zeros(256, dtype='float32')
        word_vectors = np.vstack([word_vectors, zero_vectors])

np.save('cbow/embedding', word_vectors)
