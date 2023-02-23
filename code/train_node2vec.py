# reading graph edges from bigQuery tables 
from google.cloud import bigquery
from scipy.sparse import csr_matrix
import numpy as np 
from google.cloud import storage
from google.cloud import bigquery
from pyspark.sql import *
import pandas as pd
import networkx as nx
import time 
from node2vec import *

spark = SparkSession.builder \
  .appName('data-preparation')\
  .config('spark.jars', 'gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar') \
  .getOrCreate()
  
files = ['CA-AstroPh', 'CA-GrQc',  'CA-HepPh', 'CA-HepTh' ]
# the bucket where we store the edges of each dataset in cloud storage 
BUCKET_NAME = 'rplace-bucket'
# project name in GCP 
project_name = "bigdata-project-346922"
dataset = "collaboration_data"

partitions = []
elepses = []
client = bigquery.Client()
for file in files: 
    table_id = project_name + '.' + dataset + '.' + file
    # getting tables from BigQuery 
    dataframe = client.list_rows(table_id).to_dataframe(create_bqstorage_client=True)
    # sparse matrix creation for each graph 
    rows = list(dataframe['fromId'])
    cols = list(dataframe['toId'])
    data = [1 for i in range(len(cols))]
    # we use identfiersList to map all nodes identifiers 
    # to the interval [0 - (nodes -1)]
    # in order to keep the sparse matrix small enough 
    identifiersList = [i for i in set(cols + rows)]
    identifiersList.sort()
    dic = dict()
    node = 0
    for i in identifiersList:
        dic[i] = node
        node+=1
    for k in range(len(cols)): 
        cols[k] = dic[cols[k]]
        rows[k] = dic[rows[k]]
    nodes = max(cols + rows)
    print(f"##### number of nodes {nodes} in dataset : {file}")
    m = csr_matrix((data, (rows, cols)), shape=(nodes+1, nodes+1), dtype=np.uint16)
    # m is a sparse matrix for adjecency matrix, we create then a networkx graph g 
    g = nx.from_scipy_sparse_matrix(m)
    start = time.time()
    node2vec = Node2Vec(g,dimensions=30, walk_length=32, num_walks=10,p=10, q=0.1,  workers= 8)
    print("All random walks generated for graph {}".format(file)) 
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    elepse = time.time() - start
    print("All embeddings generated for graph {}".format(file))
    print("time elepsed {}".format(elepse))
    EMBEDDING_FILENAME = "gs://"+BUCKET_NAME+"/embeddings/" + "embedding." + file
    model.wv.save_word2vec_format(EMBEDDING_FILENAME)
    elepses.append(elepse)
    
i = 0
with open("generating_time", "w") as f: 
    for file in files:
        f.write("graph : {0} - time: {1}".format(file, elepses[i]))
        i+= 1 

