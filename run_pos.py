import argparse
import spacy
from ckiptagger import WS, POS


def print_word_pos_sentence(word_sentence, pos_sentence):
    assert len(word_sentence) == len(pos_sentence)
    good = []
    bad = []
    for word, pos in zip(word_sentence, pos_sentence):
        # if pos.startswith("N"):
        if pos in ["Na", "Nb", "Nc"]:
            # print(f"{word}({pos})", end="\u3000")
            good.append((word, pos))
        else:
            bad.append((word, pos))
    print(f"看起來可以: {good}")
    print("=" * 50)
    print(f"看起來不ok: {bad}")
    print()
    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--tool",
    default="ckip",
    choices=["spacy", "ckip"],
    type=str,
)
args = parser.parse_args()

test_sentence_list = [
    "而吃東西的時候, 則靠著連著食道的右臉 。",
    "據悉, 牠的三隻眼睛裡, 中間沒有視覺能力, 其他兩個眼睛良好 。",
    "他發誓要全力幫助牠活下去, 並給牠最好的生活 。",
    "還替小貓的兩張臉各別取名字, 左臉叫 「 法蘭克 ( Frank ) 」, 右臉叫做 「 路易 ( Louie ) 」 。",
    "雙臉貓 「 Frank and Louie 」 的奇蹟故事因此聲名大噪, 受邀登上各大媒體與電視節目, 在2011年9月27日也獲得金氏世界紀錄認可為 「 全世界最長壽的雙臉貓! 」",
    "根據 《 紐約每日新聞 》 報導, 雙臉貓的女主人馬蒂 史帝夫 ( Marty Stevens ) 曾經在獸醫院工作, 12年前有人送來這隻出生僅一天大的 「 特別小貓 」, 打算將牠安樂死, 因為這種天生基因缺陷的貓咪通常活不久 。",
    # "最終東道主澳大利亞隊以14枚獎牌的成績位列該項目獎牌榜首位。",
]

if args.tool == "spacy":
    nlp = spacy.load("zh_core_web_sm")
    doc = nlp(test_sentence_list)
    for token in doc:
        print(token.text, token.pos_)

elif args.tool == "ckip":
    ws = WS("./data")
    pos = POS("./data")
    word_sentence_list = ws(test_sentence_list)  # CKIP 的輸入必定為 list
    pos_sentence_list = pos(word_sentence_list)

    for i, sentence in enumerate(test_sentence_list):
        print(f"===============第 {i} 個句子 ===============")
        print(f"'{sentence}'")
        print_word_pos_sentence(word_sentence_list[i], pos_sentence_list[i])
