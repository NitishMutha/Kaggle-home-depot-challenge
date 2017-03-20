import numpy as np
import pandas as pd
import matplotlib as plt
from src.misc import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def process_features():
    train_, test_, attributes, product_des, typos = load_data()

    alldata = pd.concat([train_, test_])

    # merge the product description
    alldata = merge_product_desp(alldata, product_des)

    # merge the brand names from attributes
    alldata = get_brand_names(alldata, attributes)

    attributes.dropna(inplace=True)

    # extract useful attributes
    bullets, bullets_count = get_properties(alldata, attributes, 'bullet')
    color, color_count = get_properties(alldata, attributes, 'color')
    material, material_count = get_properties(alldata, attributes, 'material')
    comres, comres_count = get_properties(alldata, attributes, 'commercial / residential', 'comres')
    inoutdoor, inoutdoor_count = get_properties(alldata, attributes, 'indoor/outdoor', 'inoutdoor')
    estar, estar_count = get_properties(alldata, attributes, 'energy star certified', 'estar')

    # merge the attributes
    alldata = merge_attributes(alldata, [bullets, bullets_count, color, material, comres, inoutdoor, estar])

    # drop the NA's
    NA_list = ['brand_name', 'bullet', 'bullet_count', 'color', 'material', 'flag_commercial', 'flag_residential',
               'flag_indoor', 'flag_outdoor', 'estar']
    replace_list = ['nobrand', '', 0, '', '', -1, -1, -1, -1, -1]
    alldata = remove_na(alldata, NA_list, replace_list)

    # fix typos
    alldata['search_term'] = alldata['search_term'].map(fix_typo)

    # stemming the data
    alldata['search_term'] = alldata['search_term'].map(lambda x: str_stem(x))
    alldata['product_title'] = alldata['product_title'].map(lambda x: str_stem(x))
    alldata['product_description'] = alldata['product_description'].map(lambda x: str_stem(x))
    alldata['brand_name'] = alldata['brand_name'].map(lambda x: str_stem(x))
    alldata['bullet'] = alldata['bullet'].map(lambda x: str_stem(x))
    alldata['color'] = alldata['color'].map(lambda x: str_stem(x))
    alldata['material'] = alldata['material'].map(lambda x: str_stem(x))

    # tokenization of features
    alldata['tokens_search_term'] = alldata['search_term'].map(lambda x: x.split())
    alldata['tokens_product_title'] = alldata['product_title'].map(lambda x: x.split())
    alldata['tokens_product_description'] = alldata['product_description'].map(lambda x: x.split())
    alldata['tokens_brand'] = alldata['brand_name'].map(lambda x: x.split())
    alldata['tokens_bullet'] = alldata['bullet'].map(lambda x: x.split())

    # length feature
    alldata['len_search_term'] = alldata['tokens_search_term'].map(lambda x: len(x))
    alldata['len_product_title'] = alldata['tokens_product_title'].map(lambda x: len(x))
    alldata['len_product_description'] = alldata['tokens_product_description'].map(lambda x: len(x))
    alldata['len_brand'] = alldata['tokens_brand'].map(lambda x: len(x))
    alldata['len_bullet'] = alldata['tokens_bullet'].map(lambda x: len(x))

    # find if any mention of color or materials in the search query
    alldata['match_color'] = alldata.apply(lambda x: match(x['tokens_search_term'], x['color']), axis=1).astype(
        np.float)
    alldata['match_material'] = alldata.apply(lambda x: match(x['tokens_search_term'], x['material']), axis=1).astype(
        np.float)
    alldata['flag_search_term_in_pt'] = alldata.apply(lambda x: int(x['search_term'] in x['product_title']), axis=1)
    alldata['flag_search_term_in_pd'] = alldata.apply(lambda x: int(x['search_term'] in x['product_description']), axis=1)
    alldata['flag_search_term_in_br'] = alldata.apply(lambda x: int(x['search_term'] in x['brand_name']), axis=1)
    alldata['flag_search_term_in_bl'] = alldata.apply(lambda x: int(x['search_term'] in x['bullet']), axis=1)
    alldata['flag_search_term_in_pd'] = alldata.apply(lambda x: int(x['search_term'] in x['product_description']), axis=1)

    alldata['num_search_term_in_pt'] = alldata.apply(
        lambda x: len(set(x['tokens_search_term']).intersection(set(x['tokens_product_title']))), axis=1)
    alldata['num_search_term_in_pd'] = alldata.apply(
        lambda x: len(set(x['tokens_search_term']).intersection(set(x['tokens_product_description']))), axis=1)
    alldata['num_search_term_in_br'] = alldata.apply(
        lambda x: len(set(x['tokens_search_term']).intersection(set(x['tokens_brand']))), axis=1)
    alldata['num_search_term_in_bl'] = alldata.apply(
        lambda x: len(set(x['tokens_search_term']).intersection(set(x['tokens_bullet']))), axis=1)

    alldata['ratio_search_term_in_pt'] = alldata.apply(lambda x: x['num_search_term_in_pt'] / float(x['len_search_term']), axis=1)
    alldata['ratio_search_term_in_pd'] = alldata.apply(lambda x: x['num_search_term_in_pd'] / float(x['len_search_term']), axis=1)
    alldata['ratio_search_term_in_br'] = alldata.apply(lambda x: x['num_search_term_in_br'] / float(x['len_search_term']), axis=1)
    alldata['ratio_search_term_in_bl'] = alldata.apply(lambda x: x['num_search_term_in_bl'] / float(x['len_search_term']), axis=1)

    # encode attributes
    alldata['brand_encoding'] = encode_brands(alldata)
    alldata['attr_has_material'] = encode_attributes(alldata, material)
    alldata['attr_has_color'] = encode_attributes(alldata, color)
    alldata['has_attr'] = encode_attributes(alldata, attributes)

    # distance features
    countvec = CountVectorizer(stop_words='english', max_features=1000)
    countvec.fit(
        alldata['search_term'] + ' ' + alldata['product_title'] + ' ' + alldata['product_description'] + ' ' + alldata[
            'bullet'])

    # BOW cosine similarity
    cv_of_st = countvec.transform(alldata['search_term'])
    cv_of_pt = countvec.transform(alldata['product_title'])
    cv_of_pd = countvec.transform(alldata['product_description'])
    cv_of_bl = countvec.transform(alldata['bullet'])

    cv_cos_sim_st_pt = [cosine_similarity(cv_of_st[i], cv_of_pt[i])[0][0] for i in range(cv_of_st.shape[0])]
    cv_cos_sim_st_pd = [cosine_similarity(cv_of_st[i], cv_of_pd[i])[0][0] for i in range(cv_of_st.shape[0])]
    cv_cos_sim_st_bl = [cosine_similarity(cv_of_st[i], cv_of_bl[i])[0][0] for i in range(cv_of_st.shape[0])]

    alldata['cv_cosine_sim_st_pt'] = cv_cos_sim_st_pt
    alldata['cv_cosine_sim_st_pd'] = cv_cos_sim_st_pd
    alldata['cv_cosine_sim_st_bl'] = cv_cos_sim_st_bl

    # TF-IDF
    tivec = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=1000)
    tivec.fit(
        alldata['search_term'] + ' ' + alldata['product_title'] + ' ' + alldata['product_description'] + ' ' + alldata[
            'bullet'])
    tfidf_of_st = tivec.transform(alldata['search_term'])
    tfidf_of_pt = tivec.transform(alldata['product_title'])
    tfidf_of_pd = tivec.transform(alldata['product_description'])
    tfidf_of_bl = tivec.transform(alldata['bullet'])

    tfidf_cos_sim_st_pt = [cosine_similarity(tfidf_of_st[i], tfidf_of_pt[i])[0][0] for i in range(tfidf_of_st.shape[0])]
    tfidf_cos_sim_st_pd = [cosine_similarity(tfidf_of_st[i], tfidf_of_pd[i])[0][0] for i in range(tfidf_of_st.shape[0])]
    tfidf_cos_sim_st_bl = [cosine_similarity(tfidf_of_st[i], tfidf_of_bl[i])[0][0] for i in range(tfidf_of_st.shape[0])]

    alldata['tfidf_cos_sim_st_pt'] = tfidf_cos_sim_st_pt
    alldata['tfidf_cos_sim_st_pd'] = tfidf_cos_sim_st_pd
    alldata['tfidf_cos_sim_st_bl'] = tfidf_cos_sim_st_bl

    alldata['qid'] = alldata.search_term.map(lambda x: int(np.argwhere(alldata.search_term.unique() == x).flatten()[0]))

    # save full features file
    alldata.to_csv('features_alldata.csv', index=False)

    # drops unwanted columns
    cols_drop = [
        # 'product_uid',
        'search_term',
        'product_title',
        'product_description',
        'brand_name',
        'bullet',
        'color',
        'material',
        'tokens_search_term',
        'tokens_product_title',
        'tokens_product_description',
        'tokens_brand',
        'tokens_bullet'
    ]

    final_features = alldata.drop(cols_drop, axis=1)
    final_features.to_csv('vectorised_features_final.csv', index=False)


if __name__ == '__main__':
    process_features()
