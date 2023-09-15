import pandas as pd
import numpy as np
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy.sparse import load_npz
from flask import Flask, request, render_template, redirect

# ローカル用
df_ldcc = pd.read_csv('./src/df_ramen.csv')
sim = np.load('./src/sim_matrix.npy')
X = load_npz("./src/sparse_matrix.npz")
with open("./src/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
# デプロイ用
""" df_ldcc = pd.read_csv('./df_ramen.csv')
sim = np.load('./sim_matrix.npy')
X = load_npz("./sparse_matrix.npz")
with open("./vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f) """

tokenizer = Tokenizer()
for i in range(sim.shape[0]):
    sim[i, i] = 0.0
idx = np.argsort(sim, axis=1) # 類似度高い順にソート
idx = idx[:, ::-1] # ソートの逆順に

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

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        store_keyword = request.form.get('store_keyword')
        keyword = request.form.get('keyword')
        if store_keyword:
            results = search_by_store_name(store_keyword)
        else:
            results = search_ramen(keyword)
        return render_template('result.html', results=results)
    return render_template('index.html')

def search_by_store_name(store_name):
    # 店名がdf_ldccに存在するか確認
    if store_name not in df_ldcc['store_name'].values:
        return ["申し訳ありません。店名を正確に入力してください"]
    
    keyid = df_ldcc[df_ldcc['store_name'] == store_name].index[0]
    results = []
    for i in range(10):
        store = df_ldcc['store_name'][idx[keyid, i]]
        score = df_ldcc['score'][idx[keyid, i]]
        ward = df_ldcc['ward'][idx[keyid, i]]
        similarity = sim[keyid, idx[keyid, i]]
        results.append({"name": store, 
                        "similarity": f"{similarity:.3f}", 
                        "ward": ward, 
                        "score": score})
    return results

""" def index():
    if request.method == 'POST':
        keyword = request.form.get('keyword')
        results = search_ramen(keyword)
        return render_template('result.html', results=results)
    return render_template('index.html') """

def search_ramen(keyword):
    transformed_keyword = vectorizer.transform([text2words(keyword)])
    sim1 = np.dot(transformed_keyword, X.transpose()).toarray().reshape(-1)
    idx1 = np.argsort(sim1)
    idx1 = idx1[::-1]
    #top_results = [f"{df_ldcc['store_name'][idx1[i]]} (類似度: {sim1[idx1[i]]:.3f})" for i in range(10)]
    top_results = [{"name": df_ldcc['store_name'][idx1[i]], 
                    "similarity": f"{sim1[idx1[i]]:.3f}", 
                    "ward": df_ldcc['ward'][idx1[i]], 
                    "score": df_ldcc['score'][idx1[i]]} for i in range(10)]
    return top_results

if __name__ == "__main__":
    app.run(debug=True)


