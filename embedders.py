from gensim.models import Word2Vec, Phrases
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import datetime as dt
import json
import numpy as np

class TfidfWord2VecEmbedder(object):
    
    def __init__(self, word2vec_params={}, use_bigrams=True, verbose=True):
        self._use_bigrams = use_bigrams
        self._word2vec_params = word2vec_params
        self._verbose = verbose
        
    def fit(self, collection):
        
        if self._use_bigrams:
            bigram_transformer = Phrases(collection)
            collection = bigram_transformer[collection]
            self._bigram_transformer = bigram_transformer
        
        model = Word2Vec(
            min_count=self._word2vec_params.get('min_count', 2),
            window=self._word2vec_params.get('window', 2),
            alpha=self._word2vec_params.get('alpha', 0.025),
            sample=self._word2vec_params.get('sample', 0.001),
            seed=self._word2vec_params.get('seed', 1),
            workers=self._word2vec_params.get('workers', 20),
            min_alpha=self._word2vec_params.get('min_alpha', 0.0001), 
            iter=self._word2vec_params.get('iter', 5), 
        )
        
        model.build_vocab(collection)
        
        max_epochs = self._word2vec_params.get('max_epochs', 10)

        for epoch in range(max_epochs):
            if self._verbose:
                start = dt.datetime.now()
                print('iteration {0}:'.format(epoch), end=' ')
                
            model.train(collection, total_examples=model.corpus_count, epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
            if self._verbose:
                print ('elapsed:', dt.datetime.now() - start)
            
        self._w2v_model = model
        
        dct = Dictionary(collection)
        dct.id2token = {y:x for x,y in dct.token2id.items()}
        corpus = [dct.doc2bow(line) for line in collection] 
        tfidf_model = TfidfModel(corpus)
        self._dct = dct
        self._tfidf_model = tfidf_model
        
        return self
    
    def transform(self, collection):
        if self._use_bigrams:
                collection = self._bigram_transformer[collection]
                
        vectors = []
        for text in collection:
            if len(text) < 1:
                vectors.append(np.zeros(self._w2v_model.wv.vectors.shape[1]))
                continue
                
            bow_text = self._dct.doc2bow(text)
            tfidf_text = self._tfidf_model[bow_text]
            
            if len(tfidf_text) < 1:
                vectors.append(np.zeros(self._w2v_model.wv.vectors.shape[1]))
                continue
              
            text_vecs = []
            for tokenid, weight in tfidf_text:
                token = self._dct.id2token[tokenid]
                if token in self._w2v_model.wv:
                    text_vecs.append(weight * self._w2v_model.wv[token])

            if len(text_vecs) < 1:
                vectors.append(np.zeros(self._w2v_model.wv.vectors.shape[1]))
                continue

            tfidf_vec = np.sum(text_vecs, axis=0)

            vectors.append(tfidf_vec)
            
        return np.array(vectors)
    
    def save(self, path):
        init_params = {}
        init_params['use_bigrams'] = self._use_bigrams
        init_params['verbose'] = self._verbose
        init_params['word2vec_params'] = self._word2vec_params
        
        with open(path + '/init_params.json', 'w') as f:
            json.dump(init_params, f)
        
        if self._use_bigrams:
            self._bigram_transformer.save(path + '/bigram_transformer')
            
        self._w2v_model.save(path + '/w2v_model.model')
        self._tfidf_model.save(path + '/tfidf_model.model')
        self._dct.save(path + '/gensim_dictionary')
            
    def load(path):
        with open(path + '/init_params.json', 'r') as f:
            init_params = json.load(f)
        
        embedder = TfidfWord2VecEmbedder(init_params)
        if embedder._use_bigrams:
            embedder._bigram_transformer = Phrases.load(path + '/bigram_transformer')
        
        embedder._w2v_model = Word2Vec.load(path + '/w2v_model.model')
        embedder._tfidf_model = TfidfModel.load(path + '/tfidf_model.model')
        embedder._dct = Dictionary.load(path + '/gensim_dictionary')
        
        return embedder
    
    
class MeanWord2VecEmbedder(object):
    
    def __init__(self, word2vec_params={}, use_bigrams=True, verbose=True):
        self._use_bigrams = use_bigrams
        self._word2vec_params = word2vec_params
        self._verbose = verbose
        
    def fit(self, collection):
        
        if self._use_bigrams:
            bigram_transformer = Phrases(collection)
            collection = bigram_transformer[collection]
            self._bigram_transformer = bigram_transformer
        
        model = Word2Vec(
            min_count=self._word2vec_params.get('min_count', 2),
            window=self._word2vec_params.get('window', 2),
            alpha=self._word2vec_params.get('alpha', 0.025),
            sample=self._word2vec_params.get('sample', 0.001),
            seed=self._word2vec_params.get('seed', 1),
            workers=self._word2vec_params.get('workers', 20),
            min_alpha=self._word2vec_params.get('min_alpha', 0.0001)
        )
        
        model.build_vocab(collection)
        
        max_epochs = self._word2vec_params.get('epochs', 10)

        for epoch in range(max_epochs):
            if self._verbose:
                start = dt.datetime.now()
                print('iteration {0}:'.format(epoch), end=' ')
                
            model.train(collection, total_examples=model.corpus_count, epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
            if self._verbose:
                print ('elapsed:', dt.datetime.now() - start)
            
        self._w2v_model = model
        
        return self
    
    def transform(self, collection):
        if self._use_bigrams:
                collection = self._bigram_transformer[collection]

        global_mean = np.mean(self._w2v_model.wv.vectors, axis=0)
        
        vectors = []
        for text in collection:
            
            if len(text) < 1:
                vectors.append(global_mean)
                continue

            text_vecs = [self._w2v_model.wv[t] for t in text if t in self._w2v_model.wv]

            if len(text_vecs) < 1:
                vectors.append(global_mean)
                continue

            mean_vec = np.mean(text_vecs, axis=0)

            vectors.append(mean_vec)
            
        return np.array(vectors)
    
    def save(self, path):
        if self._use_bigrams:
            self._bigram_transformer.save(path + '/bigram_transformer')
            
        self._w2v_model.save(path + '/w2v_model.model')
        
        init_params = {}
        init_params['use_bigrams'] = self._use_bigrams
        init_params['verbose'] = self._verbose
        init_params['word2vec_params'] = self._word2vec_params
        
        with open(path + '/init_params.json', 'w') as f:
            json.dump(init_params, f)
            
    def load(path):
        with open(path + '/init_params.json', 'r') as f:
            init_params = json.load(f)
        
        embedder = MeanWord2VecEmbedder(init_params)
        if embedder._use_bigrams:
            embedder._bigram_transformer = Phrases.load(path + '/bigram_transformer')
        
        embedder._w2v_model = Word2Vec.load(path + '/w2v_model.model')
        
        return embedder