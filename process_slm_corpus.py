from pathlib import Path
from tqdm import tqdm
import pickle
from prepro.tokenization import CWSTokenizer


def main(process_type):
    assert process_type in ['slm', 'raw']
    tokenizer = CWSTokenizer("prepro/vocab.txt")

    tmp_sents = []
    for data in Path("corpus").glob("*.txt"):
        print(f"Now processing {data}.")
        with open(data, "r") as f:
            sents = f.read().splitlines()

        if process_type == 'slm':
            for line in tqdm(sents):
                uscher, token, segment = tokenizer.sent_tokenize(line)
                tmp_sents.extend(token)

        elif process_type == 'raw':
            tmp_sents.extend(sents)

    if process_type == 'slm':
        processed_sents = [i for i_list in tmp_sents for i in i_list]

    elif process_type == 'raw':
        processed_sents = tmp_sents

    with open(f"corpus/{process_type}.pkl", "wb") as f: 
        pickle.dump(processed_sents, f)

if __name__ == '__main__':
    PROCESS_TYPE = 'slm'
    main(PROCESS_TYPE)