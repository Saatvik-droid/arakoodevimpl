import gensim.models as g

# doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0  # 0 = dbow; 1 = dmpv
worker_count = 1  # number of parallel processes

# pretrained word embeddings
pretrained_emb = "data/pretrained_word_embeddings.txt"  # None if use without pretrained embeddings

# input corpus
train_corpus = "data/train_docs.txt"

# output model
saved_path = "model/new.model"

# train doc2vec model
docs = g.doc2vec.TaggedLineDocument(train_corpus)
model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold,
                  workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1,
                  pretrained_emb=pretrained_emb, iter=train_epoch)

# save model
model.save(saved_path)
