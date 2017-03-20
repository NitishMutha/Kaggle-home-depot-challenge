# -*- encoding:ISO-8859-1 -*-
import numpy as np
import pandas as pd
from nltk.stem.porter import *
import requests
import time
import re
import random
from random import randint

stemmer = PorterStemmer()
random.seed(200)

stop_w = ['for', 'xbi', 'and', 'in', 'th', 'on', 'sku', 'with', 'what', 'from', 'that', 'less', 'er',
          'ing']
strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}


def load_data():
    train_ = pd.read_csv("../data/train.csv", header=0, encoding="ISO-8859-1")
    test_ = pd.read_csv("../data/test.csv", header=0, encoding="ISO-8859-1")
    attributes = pd.read_csv("../data/attributes.csv", header=0, encoding="ISO-8859-1")
    product_des = pd.read_csv("../data/product_descriptions.csv", header=0, encoding="ISO-8859-1")
    typos = pd.read_csv("../data/correctspell.csv", index_col=0).to_dict()['0']

    return train_, test_, attributes, product_des, typos


def merge_product_desp(alldata, product_des):
    return pd.merge(alldata, product_des, on="product_uid", how="left")


def get_brand_names(alldata, attributes):
    brand_names = attributes[attributes.name == "MFG Brand Name"][["product_uid", "value"]]
    brand_names = brand_names.rename(columns={"value": "brand_name"})

    alldata = pd.merge(alldata, brand_names, on="product_uid", how="left")
    alldata.brand_name.fillna("unk", inplace=True)
    return alldata


def get_properties(alldata, attributes, field, property_type=None):
    if property_type == None:
        property_type = field
    key = 'about_' + property_type
    attributes[key] = attributes['name'].str.lower().str.contains(field)
    property_ = dict()
    property_count = dict()

    for idx, row in attributes[attributes[key]].iterrows():
        pid = row['product_uid']
        value = row['value']
        property_count.setdefault(pid, 0)

        if property_type == 'comres':
            property_.setdefault(pid, [0, 0])
            if 'Commercial' in str(value):
                property_[pid][0] = 1
            if 'Residential' in str(value):
                property_[pid][1] = 1

        elif property_type == 'inoutdoor':
            property_.setdefault(pid, [0, 0])
            if 'Indoor' in str(value):
                property_[pid][0] = 1
            if 'Outdoor' in str(value):
                property_[pid][1] = 1

        elif property_type == 'estar':
            property_.setdefault(pid, 0)
            if 'Yes' in str(value):
                property_[pid] = 1
        else:
            property_.setdefault(pid, '')
            property_[pid] = property_[pid] + ' ' + str(value)

        property_count[pid] += 1

    if property_type == 'comres':
        df_property = pd.DataFrame.from_dict(property_, orient='index').reset_index().astype(np.float)
        df_property.columns = ['product_uid', 'flag_commercial', 'flag_residential']
    elif property_type == 'inoutdoor':
        df_property = pd.DataFrame.from_dict(property_, orient='index').reset_index().astype(np.float)
        df_property.columns = ['product_uid', 'flag_indoor', 'flag_outdoor']
    else:
        df_property = pd.DataFrame.from_dict(property_, orient='index').reset_index()
        df_property.columns = ['product_uid', property_type]

    df_property_count = pd.DataFrame.from_dict(property_count, orient='index').reset_index().astype(np.float)
    df_property_count.columns = ['product_uid', property_type + '_count']

    return df_property, df_property_count


def merge_attributes(alldata, attr_list):
    for atr in attr_list:
        alldata = pd.merge(alldata, atr, how='left', on='product_uid')
    return alldata


def remove_na(alldata, na_list, replace_list):
    for nal, rep in zip(na_list, replace_list):
        alldata[nal].fillna(rep, inplace=True)
    return alldata


def match(st, m):
    for w in st:
        if w in m:
            return True
    return False


def str_stem(s):
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)  # Split words with a.A
        s = s.lower()
        s = s.replace("  ", " ")
        s = s.replace(",", "")
        s = s.replace("$", " ")
        s = s.replace("?", " ")
        s = s.replace("-", " ")
        s = s.replace("//", "/")
        s = s.replace("..", ".")
        s = s.replace(" / ", " ")
        s = s.replace(" \\ ", " ")
        s = s.replace(".", " . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x ", " xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*", " xbi ")
        s = s.replace(" by ", " xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("¡ã", " degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v ", " volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  ", " ")
        s = s.replace(" . ", " ")
        try:
            s = " ".join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
            s = " ".join([stemmer.stem(z) for z in s.split(" ")])
        except Exception as inst:
            pass

        s = s.lower()
        s = s.replace("toliet", "toilet")
        s = s.replace("airconditioner", "air conditioner")
        s = s.replace("vinal", "vinyl")
        s = s.replace("vynal", "vinyl")
        s = s.replace("skill", "skil")
        s = s.replace("snowbl", "snow bl")
        s = s.replace("plexigla", "plexi gla")
        s = s.replace("rustoleum", "rust-oleum")
        s = s.replace("whirpool", "whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless", "whirlpool stainless")
        return s
    else:
        return "null"


def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]", " ", str2)
    str2 = [z for z in set(str2.split()) if len(z) > 2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word) > 3:
            s1 = []
            s1 += segmentit(word, str2, True)
            if len(s) > 1:
                s += [z for z in s1 if z not in ['er', 'ing', 's', 'less'] and len(z) > 1]
            else:
                s.append(word)
        else:
            s.append(word)
    return " ".join(s)


def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                # print(s[:-j],s[len(s)-j:])
                s = s[len(s) - j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i == len(st):
            r.append(st[i:])
    return r


def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word) >= 0:
            cnt += 1
    return cnt


def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt


typos = pd.read_csv("../data/correctspell.csv", index_col=0).to_dict()['0']


def fix_typo(s):
    if s in typos:
        return typos[s]
    else:
        return s


def encode_attributes(alldata, attr):
    pid_with_attr_material = pd.unique(attr.product_uid.ravel())
    material_encoder = {}
    for pid in pid_with_attr_material:
        material_encoder[pid] = 1
    return alldata['product_uid'].map(lambda x: material_encoder.get(x, 0)).astype(np.float)


def encode_brands(alldata):
    brands = pd.unique(alldata.brand_name.ravel())
    brand_encoder = {}
    index = 1000
    for brand in brands:
        brand_encoder[brand] = index
        index += 10
    brand_encoder['nobrand'] = 500
    return alldata['brand_name'].map(lambda x: brand_encoder.get(x, 500))


START_CHECK = "<span class=\"spell\">Showing results for</span>"
END_CHECK = "<br><span class=\"spell_orig\">Search instead for"
HTML = [("'", '&#39;'), ('"', '&quot;'), ('>', '&gt;'), ('<', '&lt;'), ('&', '&amp;')]


def spell_correction(s):
    query = '+'.join(s.split())
    time.sleep(randint(0, 2))
    r = requests.get("https://www.google.com/search?q=" + query)
    q_response = r.text
    start = q_response.find(START_CHECK)
    if (start > -1):
        start = start + len(START_CHECK)
        end = q_response.find(END_CHECK)
        search = q_response[start:end]
        search = re.sub(r'<[^>]+>', '', search)
        for code in HTML:
            search = search.replace(code[1], code[0])
        search = search[1:]
    else:
        search = s
    return search
