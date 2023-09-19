import pandas as pd
import numpy as np
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy.sparse import load_npz

df_ldcc = pd.read_csv('./src/df_ramen.csv')

# JanomeのTokenizerを初期化
tokenizer = Tokenizer()
def text2words(intext):
    tokens = tokenizer.tokenize(intext)
    outtext = ""
    for token in tokens:
        # 名詞、動詞、形容詞の基本形を取得
        if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']:
            if token.part_of_speech.split(',')[1] != '数':  # 数を含む名詞を除外
                if token.base_form == '*':  # 基本形が'*'の場合は、表層形を使用
                    outtext += ' ' + token.surface
                else:
                    outtext += ' ' + token.base_form
    return outtext

# sim_matrix.npyを読み込み
sim = np.load('./src/sim_matrix.npy')

for i in range(sim.shape[0]):
    sim[i, i] = 0.0
idx = np.argsort(sim, axis=1) # 類似度高い順にソート
idx = idx[:, ::-1] # ソートの逆順に

### 店名を入力すると類似度の高い店を表示

target_store_name = "麺屋吉左右"
keyid = df_ldcc[df_ldcc['store_name'] == target_store_name].index[0]

print(f"[{keyid:04d}] {df_ldcc['store_name'][keyid]}")

for i in range(10):  # 類似度の高い店を抽出
    store_name = df_ldcc['store_name'][idx[keyid, i]]
    score = df_ldcc['score'][idx[keyid, i]]
    ward = df_ldcc['ward'][idx[keyid, i]]
    print(f"類似度：{sim[keyid, idx[keyid, i]]:.3f} ,店名：【 {store_name} 】, {ward} , {score}")

### 任意のワードや文章を入れるとそれと類似度の高い店を表示

X = load_npz("./src/sparse_matrix.npz")
#vectorizer = TfidfVectorizer(min_df=8, max_df=0.5, use_idf=True, norm='l2')
with open("./src/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

keywords="獣のような家系ラーメン"
sim1=np.dot(vectorizer.transform([text2words(keywords)]) , X.transpose() ).toarray().reshape(-1)
idx1 = np.argsort(sim1) # 昇順ソート
idx1=idx1[::-1] # 昇順を降順に。

for i in range(10):
    store_name = df_ldcc['store_name'][idx1[i]]
    score = df_ldcc['score'][idx1[i]]
    ward = df_ldcc['ward'][idx1[i]]
    print(f"類似度: {sim1[idx1[i]]:.3f}, 店名:【 {store_name} 】, {ward} , {score}")

