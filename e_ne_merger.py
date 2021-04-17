import pandas as pd
efile="./8519771/10folds_ne_majvote_vert_corrected.csv"
nefile="./download/oc_net_256/test/final_weighted_vote.csv"

edf=pd.read_csv(efile,index_col='id')
nedf=pd.read_csv(nefile,index_col='id')

for index in nedf.index:
    if type(edf.loc[index]['rle_mask'])==float:
        #print(edf.loc[index]['rle_mask'])
        nedf.loc[index]['rle_mask']=edf.loc[index]['rle_mask']
nedf.to_csv("./download/oc_net_256/test/final_weighted_vote_with_0.871empty.csv")