import pandas as pd

r = pd.read_excel(r'c:\project\matching\dataset\roster.xlsx')
p = pd.read_excel(r'c:\project\matching\participation.xlsx')

parts = p[p['비고 O'].isin(['O', '1', 1])]['성명'].tolist()
df = r[r['성명'].isin(parts)]

print('참가자:')
print(df[['성명', '성별', '실력']])
print(f"\n남자: {len(df[df['성별']==1])}명, 여자: {len(df[df['성별']==2])}명")
