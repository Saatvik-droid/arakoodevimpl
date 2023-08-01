import os

import numpy as np
import openai
import redis
from dotenv import load_dotenv
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import PyPDF2
import gensim

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

class Embedding:
    def __init__(self, filename, query, generate_examples=False):
        create_index(300)
        self.filename = filename
        self.query = query
        self.generate_examples = generate_examples
        self.examples = []

    def _read(self):
        text = ""
        name, ext = os.path.splitext(self.filename)
        if ext == ".txt":
            with open(self.filename) as f:
                text = f.read()
        elif ext == ".pdf":
            reader = PyPDF2.PdfReader(self.filename)
            num = reader.numPages
            for i in range(num):
                text += reader.getPage(i).extractText()
        else:
            return ""
        return text

    @staticmethod
    def embed_into_db(page_content, embeddings, tag):
        pipe = r.pipeline()
        for i, (content, embedding) in enumerate(zip(page_content, embeddings)):
            # HSET
            pipe.hset(f"doc:{i}", mapping={
                "vector": embedding,
                "content": content,
                "tag": tag
            })
        pipe.execute()

    @staticmethod
    def get_embedding(text):
        model = gensim.models.doc2vec.Doc2Vec.load('model/new.model')
        return model.infer_vector(text.split())

    @staticmethod
    def convert_embedding_to_structure(embedding):
        return np.array(embedding).astype(np.float32).reshape(1, -1).tobytes()

    @staticmethod
    def format_query(tag, k):
        tag_query = f"(@tag:{{ {tag} }})=>"
        knn_query = f"[KNN {k} @vector $vec AS score]"
        query = Query(tag_query + knn_query) \
            .sort_by('score', asc=False) \
            .return_fields('id', 'score', 'content') \
            .dialect(2)
        return query

class Character(Embedding):
    def __init__(self, filename, char_count=1000, generate_examples=False):
        super().__init__(filename, self.knn_doc_query_db, generate_examples)
        self.char_count = char_count
        self.chunk_and_embed()

    def chunk(self, text):
        page_content = []
        idx = 0
        s = ""
        for ch in text:
            s += ch
            idx += 1
            if idx >= self.char_count:
                page_content.append(s)
                s = ""
                idx = 0
        return page_content

    def chunk_and_embed(self):
        text = self._read()
        page_content = self.chunk(text)
        if self.generate_examples:
            self.examples = self.generate_qa_examples(page_content)
        embeddings = []
        for content in page_content:
            embedding = self.get_embedding(content.strip())
            embeddings.append(self.convert_embedding_to_structure(embedding))
        self.embed_into_db(page_content, embeddings, "hyde")

    def knn_doc_query_db(self, query_vec, k=10):
        query = self.format_query("hyde", k)
        embedding = query_vec
        query_params = {"vec": embedding}
        ret = r.ft(INDEX_NAME).search(query, query_params).docs
        return ret

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

#
# def add_docs(document):
#     global last_id
#     last_id += 1
#     pipe = r.pipeline()
#     vec = embed(document).tobytes()
#     # HSET
#     pipe.hset(f"doc:{last_id}", mapping={
#         "vector": vec,
#         "content": document,
#         "tag": 'hyde'
#     })
#     pipe.execute()
#
#
# def retrieve(query_vector, k=1):
#     tag_query = "(@tag:{ hyde })=>"
#     knn_query = f"[KNN {k} @vector $vec AS score]"
#     redis_query = Query(tag_query + knn_query) \
#         .sort_by('score', asc=False) \
#         .return_fields('id', 'score', 'content') \
#         .dialect(2)
#     vec = query_vector
#     query_params = {"vec": vec}
#     ret = r.ft(INDEX_NAME).search(redis_query, query_params).docs
#     return ret
#
#
# def embed(document):
#     embeddings = openai.Embedding.create(input=document.strip(), model="text-embedding-ada-002")["data"][0]["embedding"]
#     embeddings = np.array(embeddings, dtype=np.float32).reshape(1, -1)
#     return embeddings


class Agent:
    def __init__(self, query, upload=False, filename=None):
        create_index(1536)
        self.query = prompt_gen.WEB_SEARCH.format(query)
        if not upload:
            self.filename = None
        else:
            self.filename = filename

    def run(self):
        c = Character("")
        print(self.query)
        result = openai.Completion.create(
            engine="text-davinci-003",
            prompt=self.query,
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

        if self.filename:
            c = Character(self.filename, 800)

        print(">Generated responses:")
        text_embeddings = []
        for idx, doc in enumerate(texts):
            print(doc.strip())
            embedding = c.get_embedding(doc.strip())
            text_embeddings.append(embedding)

        # merging response embeddings
        text_embeddings = np.array(text_embeddings)
        mean_embedding = np.mean(text_embeddings, axis=0)
        result_vector = mean_embedding.tobytes()

        query_response = c.knn_doc_query_db(result_vector, 2)
        print(query_response)
        print(">Retrieved documents:")
        for res in query_response:
            print(res.content.strip())


if __name__ == "__main__":
    a = Agent("Provisioning norms in respect of all cases of fraud", upload=True, filename="data/data.pdf")
    a.run()