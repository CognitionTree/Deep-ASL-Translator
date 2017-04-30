import gensim
from numpy import *
from nltk.tokenize import word_tokenize
from keras.models import model_from_json

trained_models_dir = 'trained_models/'
model_path_save = 'trained_models/word2vec_model_100'
sentence_size = 50

def tokenize_one_sentence(sentence):
	sentence = word_tokenize(sentence)
	tokenized_sentence = []
	for word in sentence:
		word = word.lower()
		if word.isalnum():
			tokenized_sentence.append(word)
	return tokenized_sentence

def tokenize_all_sentences(sentences):
	sentence_array = []
	i = 0
	for sentence in sentences:
		print i, " out of ", len(sentences)	
		tokenized_sentence = tokenize_one_sentence(sentence)
		sentence_array.append(tokenized_sentence)
		i += 1
	return sentence_array
		
def get_sentence_encoding(sentence, model_path = model_path_save, sentence_size = sentence_size):
	model = gensim.models.Word2Vec.load(model_path)
	sentence = ' '.join(tokenize_one_sentence(sentence))
	sentence_emb = model[sentence.split()]
	
	n_words = len(sentence_emb)
	embedding_size = len(sentence_emb[0])
	padding_size = sentence_size - n_words
	padded_sentence = sentence_emb
	
	if padding_size != 0:
		padding = zeros((padding_size, embedding_size))
		padded_sentence = append(padding, sentence_emb, axis=0)
	return padded_sentence
	
def save_keras_mode(name, model):
	model_json = model.to_json()
	with open(trained_models_dir+ name + '.json', 'w') as json_file:
		json_file.write(model_json)
	
	model.save_weights(trained_models_dir + name + ".h5")

def time_sum(full_emb):
	n_words, emb_size = full_emb.shape
	full_emb /= (n_words*1.0)
	emb = zeros(emb_size)
	
	for i in range(n_words):
		emb += (i+1)*full_emb[i]
	
	return emb
