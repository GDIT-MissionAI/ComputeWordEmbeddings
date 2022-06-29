import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize
import pickle
from sentence_transformers import SentenceTransformer

# Does sentence transformer support reloaded models? We'll need to find out.
# Also, SentenceTransformer is deprecated, but I don't have time to replace model/redo code.
# Need to replace SentenceTransformer with https://www.sbert.net/docs/pretrained_models.html
# loadedTransformer = SentenceTransformer.from_pretrained("model/")

def lambda_handler(event, context):
  
  

base_document_uri = "s3://<mybucket/mykey"

# This will download and load the pretrained model offered by UKPLab.
# Need to see if we can use the preloaded model, which is loaded in lambda container rather than downloading on every execution.
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Although it is not explicitly stated in the official document of sentence transformer, the original BERT is meant for a shorter sentence. We will feed the model by sentences instead of the whole documents.

# read document from s3 
base_document = "read base_document_uri into string"

# convert document to vector of word embeddings
sentences = sent_tokenize(base_document)
base_embeddings_sentences = model.encode(sentences)
base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)
serialized_embedding = pickle.dumps(base_embeddings)

# write to dynamodb
