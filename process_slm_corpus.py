from pathlib import Path
from tqdm import tqdm
import pickle
from prepro.tokenization import CWSTokenizer

tokenizer = CWSTokenizer("prepro/vocab.txt")

tmp_sents = []
for data in Path("corpus").glob("*.txt"):
# for data in ["corpus/pku_unsegmented.txt"]:
    print(f"Now processing {data}.")
    with open(data, "r") as f:
        sents = f.read().splitlines()

    for line in tqdm(sents):
        uscher, token, segment = tokenizer.sent_tokenize(line)
        tmp_sents.extend(token)

processed_sents = [i for i_list in tmp_sents for i in i_list]
with open("corpus/processesed.pkl", "wb") as f: 
    pickle.dump(processed_sents, f)
