import pandas as pd

# with open(r'./procedure.xlsx', 'rb') as f:
#     data = pd.read_excel(f)
#     print(data)
#     data = data[['ID', 'six_bits_code']].rename({'six_bits_code': 'Code'}, axis=1)
#     data.to_excel(r'./pro_in.xlsx', index=False)

# with open(r'./atc.xlsx', 'rb') as f:
#     data = pd.read_excel(f)
#     print(data)
#     data = data[['ID', 'atc_code']].rename({'atc_code': 'Code'}, axis=1)
#     data.to_excel(r'./atc_in.xlsx', index=False)

# with open(r'./lab.xlsx', 'rb') as f:
#     data = pd.read_excel(f)
#     print(data)
#     data = data[['ID', 'Unified_Assay_Item_Name']].rename({'Unified_Assay_Item_Name': 'Code'}, axis=1)
#     data.to_excel(r'./lab_in.xlsx', index=False)

# with open(r'./Unified_Assay_Item_Name_new.xlsx', 'rb') as f:
#     data = pd.read_excel(f)
#     print(data)
#     data = data[['Unified_Assay_Item_Name_new']].rename({'Unified_Assay_Item_Name_new': 'Code'}, axis=1)
#     data = pd.concat([data['Code']+'_high', data['Code']+'_low'], axis=0)
#     data = data.replace('', float('nan')).dropna()
#     data.to_excel(r'./lab_list.xlsx', index=False)

# with open(r'./diag_CC.xlsx', 'rb') as f:
#     data = pd.read_excel(f)
#     print(data)
#     data = data[['ID', 'icd10', 'type']].rename({'icd10': 'Code', 'type': 'label'}, axis=1)
#     data = data.loc[data['label'].notna()]
#     data[['ID', 'Code']].to_excel(r'./diag_out.xlsx', index=False)

with open(r'./pro_in.xlsx', 'rb') as f:
    pro_in = pd.read_excel(f).dropna()
    pro_in = pro_in.groupby('ID').aggregate(','.join).reset_index()

with open(r'./atc_in.xlsx', 'rb') as f:
    atc_in = pd.read_excel(f).dropna()
    atc_in = atc_in.groupby('ID').aggregate(','.join).reset_index()

with open(r'./lab_in.xlsx', 'rb') as f:
    lab_in = pd.read_excel(f).dropna()
    lab_in = lab_in.groupby('ID').aggregate(','.join).reset_index()

with open(r'./diag_out.xlsx', 'rb') as f:
    diag_out = pd.read_excel(f).dropna()
    diag_out = diag_out.groupby('ID').aggregate(
        ','.join).apply(lambda x: x.replace('.', '')).reset_index()

dataset = pro_in.merge(atc_in, on='ID', how='outer',
                       suffixes=('_pro', '_atc')).merge(
                           lab_in, on='ID', how='outer',
                           suffixes=('', '_lab')).merge(
                               diag_out,
                               on='ID',
                               how='outer',
                               suffixes=('_lab',
                                         '_diag')).dropna(subset=['Code_diag'],
                                                          axis=0)
dataset = dataset.apply(lambda s: s.fillna({i: [] for i in dataset.index}))
dataset.to_excel(r'./dataset.xlsx', index=False)
