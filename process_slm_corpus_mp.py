from time import time
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pickle
from prepro.tokenization import CWSTokenizer


s_time = time()

tokenizer = CWSTokenizer("prepro/vocab.txt")

lines = []
tmp_sents = []
for data in Path("corpus").glob("*.txt"):
# for data in ["corpus/pku_unsegmented.txt"]:
    print(f"Now processing {data}.")
    lines.extend(open(data, "r").readlines())

out = process_map(tokenizer.sent_tokenize, lines, max_workers=20, chunksize=1500)
tmp_sents = [i[1] for i in out]
processed_sents = [i for i_list in tmp_sents for i in i_list]

with open("corpus/processesed_multi.pkl", "wb") as f: 
    pickle.dump(processed_sents, f)

print(f'Process time: {time() - s_time}')