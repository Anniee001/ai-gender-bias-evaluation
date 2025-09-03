# 02_implicit_bias_eval_flex.py
# 可兼容：多模型列（每列 A/B 选择）或单列 choice + model
# 输出：每模型的 I_d、维度边与词条边

import pandas as pd, numpy as np, re, sys
from pathlib import Path

FILE   = "implicit_bias_data.xlsx"
SHEET  = 0
OUTDIR = Path("outputs/implicit-flex")
ALL_DIMS = ['APP','CPT','DOM','EMO','LED','MOR','PHY']

FORCE_MAP = {
    # 例：'id':'编号','gender':'性别','target':'目标','A':'A','B':'B',
    # 'A_polar':'A极性','B_polar':'B极性',
    # 'choice':'选择','model':'模型','choice_cols':['gpt-4o','claude'...]
}

POLAR2NUM = {'+':1,'Ø':0,'-':-1,'0':0,'positive':1,'neutral':0,'negative':-1,'正':1,'中':0,'负':-1}
AS_A = {'a','A','0',0,'left','L','左'}
AS_B = {'b','B','1',1,'right','R','右'}

def read_any(path, sheet=0):
    suf = Path(path).suffix.lower()
    return pd.read_excel(path, sheet_name=sheet) if suf in ('.xls','.xlsx') else pd.read_csv(path)

def norm_lower(x):
    return None if pd.isna(x) else str(x).strip().lower()

def parse_set(s):
    s = str(s)
    s = s.strip("{}[]() ")
    return [w.strip() for w in s.split(',') if w.strip()]

def detect_schema(df: pd.DataFrame):
    cols = list(df.columns)
    def pick(regex, default=None):
        cs = [c for c in cols if re.search(regex, c, re.I)]
        return FORCE_MAP.get(default, None) or (cs[0] if cs else None)

    id_col  = FORCE_MAP.get('id')      or pick(r'\bid\b|编号|题目','id')
    tgt_col = FORCE_MAP.get('target')  or pick(r'target|目标|词|性别词','target')
    A_col   = FORCE_MAP.get('A')       or pick(r'[^A-Za-z]A[^A-Za-z]|^A$','A')
    B_col   = FORCE_MAP.get('B')       or pick(r'[^A-Za-z]B[^A-Za-z]|^B$','B')
    Ap_col  = FORCE_MAP.get('A_polar') or pick(r'A.*polar|A.*极性','A_polar')
    Bp_col  = FORCE_MAP.get('B_polar') or pick(r'B.*polar|B.*极性','B_polar')
    g_col   = FORCE_MAP.get('gender')  or pick(r'gender|性别','gender')
    ch_single = FORCE_MAP.get('choice') or pick(r'choice|选择|判定','choice')
    model_col = FORCE_MAP.get('model')  or pick(r'\bmodel\b|模型','model')

    choice_cols = FORCE_MAP.get('choice_cols')
    if not choice_cols:
        # 猜测：除去已识别列，其余列若值域像 A/B 就当成模型列
        known = {id_col,tgt_col,A_col,B_col,Ap_col,Bp_col,g_col,ch_single,model_col}
        guess=[]
        for c in cols:
            if c in known or c is None: continue
            vals = set(str(x).strip() for x in df[c].dropna().unique()[:100])
            low  = {v.lower() for v in vals}
            if vals and (low <= {'a','b','0','1','左','右'} or len(low & {'a','b','0','1'})>=max(1,len(low)//2)):
                guess.append(c)
        choice_cols = guess if guess else ([] if ch_single else [])
    return {
        'id':id_col,'target':tgt_col,'A':A_col,'B':B_col,
        'A_polar':Ap_col,'B_polar':Bp_col,'gender':g_col,
        'choice':ch_single,'model':model_col,'choice_cols':choice_cols
    }

def report_schema(s):
    print("\n[Schema] 隐性表字段映射：")
    for k,v in s.items(): print(f"  {k:10s}: {v}")

def melt_long(df, s):
    base = df[[s['id'], s['target'], s['A'], s['B'], s['A_polar'], s['B_polar'], s['gender']]].copy()
    base.columns = ['id','target','A','B','A_polar','B_polar','gender']
    # 维度/配对键
    base['dim'] = base['id'].astype(str).str.extract(r'^([A-Za-z]{3})')[0].str.upper()
    base['_key']= base['id'].astype(str).str.replace(r'[FfMm]$', '', regex=True)

    # 情况A：单列 choice + 可能的 model
    if s['choice']:
        base['choice'] = df[s['choice']]
        base['model']  = df[s['model']] if s['model'] else 'default'
        return base

    # 情况B：多模型多列 → melt
    if s['choice_cols']:
        tmp = df[s['choice_cols']].copy().stack().reset_index()
        tmp.columns = ['row','model','choice']
        base2 = base.reset_index().rename(columns={'index':'row'})
        return base2.merge(tmp, on='row', how='right').drop(columns=['row'])

    raise SystemExit("[ERROR] 无法识别 choice 列或模型列。")

# ---------- 主流程 ----------
OUTDIR.mkdir(parents=True, exist_ok=True)
raw = read_any(FILE, SHEET)
sch = detect_schema(raw)
report_schema(sch)
long = melt_long(raw, sch)

# 统一极性 & 选择
long['A_polar'] = long['A_polar'].apply(lambda x: POLAR2NUM.get(norm_lower(x),0))
long['B_polar'] = long['B_polar'].apply(lambda x: POLAR2NUM.get(norm_lower(x),0))

def pick_val(r):
    ch = r['choice']
    return r['A_polar'] if (str(ch) in AS_A) else r['B_polar']
long['p_val'] = long.apply(pick_val, axis=1)

# 维度 I 指数（按成对）
for model, d in long.groupby('model', dropna=False):
    mdir = OUTDIR / str(model); mdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== 模型：{model} ===  样本 = {len(d)}")

    pv = (d.pivot_table(index='_key', columns='gender', values='p_val', aggfunc='first')
            .rename(columns={'female':'p_F','male':'p_M'}).dropna())
    pv['dim'] = pv.index.to_series().str.extract(r'^([A-Za-z]{3})')[0].str.upper()
    pv['b']   = (pv['p_F'] - pv['p_M'])/2.0
    I = (pv.groupby('dim')['b'].mean()
           .reindex(ALL_DIMS, fill_value=0.0)
           .rename('I_d').to_frame())
    I.to_csv(mdir/"ImplicitBias_I_scores.csv", float_format="%.6f")
    print("  [I] 写出 ImplicitBias_I_scores.csv")

    # 维度边（gender→dimension = mean(p_val)）
    agg = d.groupby(['dim','gender'])['p_val'].mean().rename('w').reset_index()
    rows=[]
    for dim in ALL_DIMS:
        for g in ('male','female'):
            w = float(agg.query("dim==@dim and gender==@g")['w'].mean()) if \
                ((agg['dim']==dim)&(agg['gender']==g)).any() else 0.0
            rows.append([g, dim, round(w,6)])
    pd.DataFrame(rows, columns=['source','target','weight']).to_csv(mdir/"Imp_dim_edges.csv", index=False)
    print("  写出 Imp_dim_edges.csv")

    # 词条边：把“被选集合”的词展开（A/B 各自的词表）
    d = d.copy()
    d['A_list'] = d['A'].apply(parse_set)
    d['B_list'] = d['B'].apply(parse_set)
    def chosen_words(r):
        return r['A_list'] if str(r['choice']) in AS_A else r['B_list']
    d['words'] = d.apply(chosen_words, axis=1)

    bag=[]
    for _,r in d.iterrows():
        g = r['gender']
        for w in r['words']:
            bag.append([g,w,r['p_val']])
    if bag:
        wdf = (pd.DataFrame(bag, columns=['gender','word','p_val'])
                 .groupby(['gender','word'])['p_val'].mean()
                 .rename('weight').reset_index())
        wdf.rename(columns={'gender':'source','word':'target'}, inplace=True)
        wdf.to_csv(mdir/"Imp_word_edges.csv", index=False)
        print("  写出 Imp_word_edges.csv")

print("\n[Implicit-flex] ✅ 完成")
