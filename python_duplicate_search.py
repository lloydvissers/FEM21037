import pandas as pd
import re
import random
import csv
import itertools
from sklearn.model_selection import train_test_split 
from joblib import Parallel, delayed
import datetime
import matplotlib.pyplot as plt
import numpy as np
    
def add_brand(data):
    data['brand'] = ''
    brandlist = list(csv.reader(open("brand_list.csv", "r")))
    for i in range(len(data)):
        title_elements = re.findall("[a-zA-Z0-9]+", data['title'][i].lower())
        for brand in brandlist:
            if brand[0].lower() in title_elements:
                data.at[i,'brand'] = brand[0].lower()
    return data

def get_tot_pairs(data):
    pairs = itertools.combinations(data.index.tolist(), 2)
    k = 0
    for pair in pairs:
        k+=1
    return k

def get_pap(data):
    output_pair = set()
    pairs = itertools.combinations(data.index.tolist(), 2)
    for pair in pairs:
        output_pair.add(tuple(pair))
    return output_pair

def map_binary_1(mw_all, mw_products):
    df = pd.DataFrame(0, index=mw_all, columns=range(len(mw_products)))
    for word in df.index:
        for product_ID in df.columns:
            if word in list(mw_products[product_ID]):
                df[product_ID][word] = 1
    return df     

def create_bands(signature_matrix,r):   
    signature_M = []
    for i in range(0,len(signature_matrix),r):
        row = []
        for j in range(len(signature_matrix.columns)):
            row.append(signature_matrix[j][i:i+r])
        signature_M.append(row)
    return signature_M

def LSH(band_matrix):
    kp_list = []
    for band in band_matrix:
        for i in range(len(band)):
            for k in range(len(band)):
                if band[i].tolist() == band[k].tolist() and i!=k:
                    if [i,k] not in kp_list and [k,i] not in kp_list:
                        kp_list.append([i,k])
                    break
    return kp_list

def jaccard(x,y):
    return len(x.intersection(y)) / len(x.union(y))

def evaluate(data,candidatepairs, pairs_estimate, pair_real):
    pairs_all_possible  = get_pap(data)
    pair_real           = get_realpairs(data)

    tp = len(pair_real.intersection(pairs_estimate))
    fp = len(pairs_estimate.intersection((pairs_all_possible-pair_real)))
    fn = len(pair_real.intersection((pairs_all_possible-pairs_estimate)))
    
    if tp == 0:
        tp = 0.0001
    if fp == 0:
        fp = 0.0001  
    if fn == 0:
        fn = 0.0001   
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    
    Fc = len(candidatepairs)/len(pairs_all_possible)
    Df = tp
    Dn = len(pair_real)
    Nc = len(candidatepairs)
    
    #Df total amount of duplicates found
    #Dn real duplicate amount
    #Nc number of comparisons made
    
    
    if Df == 0:
        Df = 0.0001
    if Dn == 0:
        Dn = 0.0001
    if Nc == 0:
        Nc = 0.0001
    
    PQ = Df/Nc
    PC = Df/Dn
    F1 = (2*recall*precision)/(recall+precision)
    F1s = (2*PQ*PC)/(PQ+PC)
    return [Fc,PQ,PC,F1,F1s]

def get_clean_data(data):      
    standard_inch= ['-inch','inch', 'inches', '”', '-inch',' inch', 'inch']
    standard_hz = ['-hz', 'hertz', 'hz']
    mw_per_product = []
    
    for title in data['title']:
        output_product_mw = set()
        title = title.replace('"','inch').lower()       
        words_title = re.findall("[a-zA-Z0-9]+", title)
        
        for element in standard_inch:
           title =  title.replace(element, "inch")
           
        for element in standard_hz:
            title =  title.replace(element, "hz")
        
        for word in words_title:
            if word.isalnum() and not word.isdigit() and not word.isalpha():
                if word not in output_product_mw:
                    output_product_mw.add(word)
        mw_per_product.append(output_product_mw)
    return mw_per_product

def get_all(mw_products):
    mw_all = mw_products[0]
    for set in mw_products:
        mw_all = mw_all.union(set)
    return mw_all

def estimate(data, candidatepairs, mw_products, r,n,t):   
    pair_estimates = set()
    for pair in candidatepairs:
        if (jaccard(mw_products[pair[0]],mw_products[pair[1]]) >= t):
            pair_estimates.add(tuple(pair))
    return pair_estimates        

def get_realpairs(data_XYZ):
    pairs = set()
    for i in range(len((data_XYZ['modelID']))):
        for j in range(len((data_XYZ['modelID']))):
            if (data_XYZ['modelID'].iloc[i]) == (data_XYZ['modelID'].iloc[j]) and j>i:
                pairs.add(tuple([i,j]))
    return pairs

def minhash(n, binary_vector_matrix):
    random.seed(1)
    a = random.choices(range(1,5000), k=n)
    b = random.choices(range(1,5000), k=n)
    
    sigmat = pd.DataFrame(999999, index=range(n),columns=binary_vector_matrix.columns)
    
    for row in range(len(binary_vector_matrix)):
        list_hashes =[]
        for i in range(n):
            hash = (a[i]*row + b[i]) % n
            list_hashes.append(hash)
            
        for column in binary_vector_matrix.columns:
            if binary_vector_matrix[column][row] == 1:
                for i in range(n):
                    if list_hashes[i] < sigmat[column][i]:
                        sigmat[column][i] = list_hashes[i]       
    
    return sigmat

def get_tp(A,B):
    k=0
    for element in A:
        if element in B:
            k += 1
    return k

def plot(r_range, results):
   #PLOT: FractionComparisons x PairQuality
   for r in r_range:
       y = results[results['r'] == r]['train_PQ']
       x = results[results['r'] == r]['train_FC']
       label = str(r)
       plt.plot(x, y, label = label)
       plt.savefig('plots/'+str(r)+' FCxPQ.png')
       plt.show()

   #PLOT: FractionComparisons x PairCompleteness
   for r in r_range:
       y = results[results['r'] == r]['train_PC']
       x = results[results['r'] == r]['train_FC']
       label = str(r)
       plt.plot(x, y, label = label)
       plt.savefig('plots/'+str(r)+' FCxPC.png')
       plt.show()
       
   #PLOT: FractionComparisons x F1
   for r in r_range:
       y = results[results['r'] == r]['train_PQ']
       x = results[results['r'] == r]['train_F1']
       label = str(r)
       plt.plot(x, y, label = label)
       plt.savefig('plots/'+str(r)+' FCxF1.png')
       plt.show()
       
   for r in r_range:
       y = results[results['r'] == r]['train_PQ']
       x = results[results['r'] == r]['train_F1s']
       label = str(r)
       plt.plot(x, y, label = label)
       plt.savefig('plots/'+str(r)+' FCxF1s.png')
       plt.show()

def execute_duplicatesearch(data, r,t):
    data_c = get_clean_data(data)
    binary_vector_matrix = map_binary_1(get_all(data_c), data_c)
    n = int(10*round(len(binary_vector_matrix)/2/10))
    signature_matrix    = minhash(n, binary_vector_matrix)   
    bands               = create_bands(signature_matrix, r)
    candidatepairs      = LSH(bands)  
    pair_estimates      = estimate(data, candidatepairs, data_c, r,n,t)    
    pair_real           = get_realpairs(data)
    
    return n/r, evaluate(data,candidatepairs, pair_estimates, pair_real)

def worker(k,r,t):
    data = pd.read_csv('tv_matrix.csv')
    data = add_brand(data)
    data_train, data_test   = train_test_split(data, test_size=0.4, random_state=k)
    results_train           = execute_duplicatesearch(data_train, r,t)
    results_test            = execute_duplicatesearch(data_test, r,t)
    return k, r, t, results_train, results_test

def run():
    k_range = range(5)
    r_range = [1,2,3,4,5,6,7,8,9,10]
    t_range = list(np.arange(0,1,0.05))
    
    results = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
            delayed(worker)(k,r,t)
            for k in k_range
            for r in r_range
            for t in t_range)
    
    df = pd.DataFrame(columns=['k','r','t', 'train_b','train_FC','train_PQ', 'train_PC', 'train_F1', 'train_F1s','test_b', 'test_FC', 'test_PQ', 'test_PC', 'test_F1', 'test_F1s'], index=range(len(results)))
    for i in range(len(results)):
        df["k"][i] = results[i][0]
        df["r"][i] = results[i][1]
        df["t"][i] = results[i][2]
    
        df["train_b"][i] = results[i][3][0]
        df['train_FC'][i] = results[i][3][1][0]
        df['train_PQ'][i] = results[i][3][1][1]
        df['train_PC'][i] = results[i][3][1][2]
        df['train_F1'][i] = results[i][3][1][3]
        df['train_F1s'][i] = results[i][3][1][4]
        df["test_b"][i]   = results[i][4][0]
        df['test_FC'][i] = results[i][4][1][0]
        df['test_PQ'][i] = results[i][4][1][1]
        df['test_PC'][i] = results[i][4][1][2]
        df['test_F1'][i] = results[i][4][1][3]
        df['test_F1s'][i] = results[i][4][1][4]
    
    df = df.groupby(['r','t']).mean()
    filename = '221209_run' + str(datetime.datetime.now().minute)
    df.to_csv('results/' +filename+ '.csv')
    results = pd.read_csv('results/' +filename+ '.csv')
    plot(r_range, results)
    print(df)
     
run()



