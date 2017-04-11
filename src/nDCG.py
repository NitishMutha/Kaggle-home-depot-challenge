import pandas as pd
import numpy as np

def create_ndcg(filename)

    gt_df = pd.read_csv('../data/solution.csv')
    gt_df.rename(index=str, columns={'relevance': 'gt_relevance'}, inplace='TRUE')
    ff_df = pd.read_csv('../data/vectorised_features_final.csv')
    bf_df = pd.read_csv(filename)

    in_df = ff_df[['id','qid','relevance']]
    in_df = in_df.merge(gt_df, how='left', on='id')
    del(gt_df)
    del(ff_df)
    in_df.loc[74067:, 'relevance'] = in_df.loc[74067:, 'gt_relevance']
    in_df.loc[:74066, 'Usage'] = 'Train'
    in_df.loc[:74066, 'gt_relevance'] = in_df.loc[:74066, 'relevance']
    in_df=in_df.drop('relevance', axis=1)

    bf_df = bf_df.merge(in_df, how='left', on='id')
    del(in_df)
    bf_df = bf_df[bf_df['Usage'] != 'Ignored']
    bf_df.sort_values(['qid','relevance'],ascending=[1,0] ,inplace=True)
    bf_df = bf_df.reset_index(drop=True)



    ndcg_k = 10
    grouped = bf_df.groupby('qid')
    t_dcg = 0
    for name, group in grouped:

        gt_sort = sorted(group['gt_relevance'], reverse=True)
        IDCGk = 0
        for r, x in enumerate(gt_sort[:ndcg_k]):
            IDCGk += np.power(2,x) / np.log2(r + 2)
            ranked = group['gt_relevance']    
            DCGk = 0
        for r, x in enumerate(ranked[:ndcg_k]):
            DCGk += np.power(2,x) / np.log2(r + 2)

        t_dcg += DCGk/IDCGk
        last = name

    print(t_dcg/last))
    return

if __name__ == '__main__':
    create_ndcg(filename = '../data/output/Results/submission_bf.csv')
