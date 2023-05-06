# Storage - Update to include query but primarily use logic to store the outputs as .txt files initially.

import os
from dotenv import load_dotenv
import pinecone
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
from tqdm.auto import tqdm  # this is our progress bar

from datasets import load_dataset


load_dotenv()
openai_api_key=os.environ.get("OPENAI_API_KEY")

def embed_upsert(data, index_name='mlqai'):
    openai_api_key=os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()

    # initialize pinecone
    #https://docs.pinecone.io/docs/openai

    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.environ.get("PINECONE_ENV")  # next to api key in console
    )
   
   
#index retrieved from function
    if index_name not in pinecone.list_indexes():
        raise ValueError(
            f"No '{index_name}' index exists. You must create the index before "
            "running this notebook. Please refer to the walkthrough at "
            "'github.com/pinecone-io/examples'."  # TODO add full link
        )

    index = pinecone.Index(index_name)
    
    

    batch_size = 32  # process everything in batches of 32
    for i in tqdm(range(0, len(data['text']), batch_size)):
        # set end position of batch
        i_end = min(i+batch_size, len(data['text']))
        # get batch of lines and IDs
        lines_batch = data['text'][i: i+batch_size]
        ids_batch = [str(n) for n in range(i, i_end)]
        # create embeddings
        MODEL = "text-embedding-ada-002"
        
        res = openai.Embedding.create(input=lines_batch, engine=MODEL)
        embeds = [record['embedding'] for record in res['data']]
        # prep metadata and upsert batch
        meta = [{'text': line} for line in lines_batch]
        to_upsert = zip(ids_batch, embeds, meta)
        # upsert to Pinecone
        index.upsert(vectors=list(to_upsert))
    
    return print(to_upsert)


def query_vec(query, index=pinecone.Index('mlqai')):
    
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.environ.get("PINECONE_ENV")  # next to api key in console
    )
    index=index
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    
    
    MODEL = "text-embedding-ada-002"
   
    xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
    res = index.query([xq], top_k=3, include_metadata=True, )
    
    for match in res['matches']:
        ls = f"{match['score']:.2f}: {match['metadata']['text']}"
    
    return print(ls)


# load the first 1K rows of the TREC dataset
trec = load_dataset('trec', split='train[:1000]')
query = "What caused the 1929 Great Depression?"

#embed_upsert(trec)
query_vec(query)



'''
res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=MODEL
)'''


