import os

import numpy as np
import openai
import redis
from dotenv import load_dotenv
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from prompt import PromptGen

load_dotenv()
prompt_gen = PromptGen()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

r = redis.Redis(
    host='redis-18848.c212.ap-south-1-1.ec2.cloud.redislabs.com',
    port=18848,
    password=os.getenv("REDIS_PASSWORD"))

INDEX_NAME = "hyde"  # Vector Index Name
DOC_PREFIX = "doc:"  # RediSearch Key Prefix for the Index

last_id = -1


def create_index(vector_dimensions: int):
    try:
        r.ft(INDEX_NAME).dropindex(delete_documents=True)
    except:
        pass

    # schema
    schema = (
        TagField("tag"),
        VectorField("vector",  # Vector Field Name
                    "FLAT", {  # Vector Index Type: FLAT or HNSW
                        "TYPE": "FLOAT32",  # FLOAT32 or FLOAT64
                        "DIM": vector_dimensions,  # Number of Vector Dimensions
                        "DISTANCE_METRIC": "COSINE",  # Vector Search Distance Metric
                    }
                    ),
    )

    # index Definition
    definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)

    # create Index
    r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)


def add_docs(document):
    global last_id
    last_id += 1
    pipe = r.pipeline()
    vec = embed(document).tobytes()
    # HSET
    pipe.hset(f"doc:{last_id}", mapping={
        "vector": vec,
        "content": document,
        "tag": 'chat_history'
    })
    pipe.execute()


def retrieve(query_vector, k=1):
    tag_query = "(@tag:{ chat_history })=>"
    knn_query = f"[KNN {k} @vector $vec AS score]"
    redis_query = Query(tag_query + knn_query) \
        .sort_by('score', asc=False) \
        .return_fields('id', 'score', 'content') \
        .dialect(2)
    vec = query_vector
    query_params = {"vec": vec}
    ret = r.ft(INDEX_NAME).search(redis_query, query_params).docs
    return ret


def embed(document):
    embeddings = openai.Embedding.create(input=document.strip(), model="text-embedding-ada-002")["data"][0]["embedding"]
    embeddings = np.array(embeddings, dtype=np.float32).reshape(1, -1)
    return embeddings


create_index(1536)

query = prompt_gen.WEB_SEARCH.format("how long does it take to remove wisdom tooth")
result = openai.Completion.create(
    engine="text-davinci-003",
    prompt=query,
    temperature=0.7,
    n=8,
    max_tokens=512,
    top_p=1,
    stop=['\n\n\n'],
    logprobs=1
)

# logarithmic probability sorting
to_return = []
for _, val in enumerate(result['choices']):
    text = val['text']
    logprob = sum(val['logprobs']['token_logprobs'])
    to_return.append((text, logprob))
texts = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]

for doc in texts:
    add_docs(doc)

print(">Generated responses:")
text_embeddings = []
for idx, doc in enumerate(texts):
    print(doc.strip())
    embedding = embed(doc.strip())
    text_embeddings.append(embedding)

# merging response embeddings
text_embeddings = np.array(text_embeddings)
mean_embedding = np.mean(text_embeddings, axis=0)
result_vector = mean_embedding.tobytes()

query_response = retrieve(result_vector, 2)
print(">Retrieved documents:")
for res in query_response:
    print(res.content.strip())
