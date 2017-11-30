from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
from operator import itemgetter
import numpy as np
import gzip
import string
import random


PATH_EMB = "./askubuntu/vector/vectors_pruned.200.txt.gz"
PATH_QUESTION_CORPUS = "./askubuntu/text_tokenized.txt.gz"
PATH_DEV = "./askubuntu/dev.txt"
PATH_TEST = "./askubuntu/test.txt"

TEXTMAX_LENGTH = 100 #int or None


def vectorizeData(filename, binary_count, vocab, stop_words_, ngram_range_, max_df_, min_df_):

	vectorizer = TfidfVectorizer(input ='content', binary=binary_count, vocabulary=vocab, stop_words= stop_words_, ngram_range=ngram_range_, max_df=max_df_, min_df= min_df_)

	data = {}
	with gzip.open(filename) as gfile:
		for row in gfile:			
			row_arr = row.split()

			row_string = string.join(row_arr[1:TEXTMAX_LENGTH + 1])

			data[row_arr[0]] = row_string
	
	vectorizer.fit(data.values())
	
	return vectorizer, data


def buildDictionary():
	vocabulary = []
	with gzip.open(PATH_EMB) as gfile:
		for line in gfile:
			word = line.split()[0]
			vocabulary.append(word)

	return vocabulary


def createVectorLabelTuples(all_q, questions_dict, q, list_of_pos, vectorizer):
	vector_label_list = []

	for item in all_q: #item is a number referring to a question
		candidate = item

		if candidate not in questions_dict:
			while True:
				candidate = random.choice(all_question_numbers)
				if candidate not in set().union(all_q, [q]):
					break

		vector = vectorizer.transform([questions_dict[candidate]]).toarray()
		#vector[0]?
		label = 1 if candidate in list_of_pos else 0
		vector_label_list.append((vector[0], label))

	return vector_label_list



def updateScores(scores_list, sum_av_prec, sum_ranks, num_samples, top_5, top_1):

	count = 0.0
	last_index = -1
	sum_prec = 0.0
	similar_indices = []
	flag = 0

	similars_total_count = [1 for x in scores_list if x[1] == 1]
	similars_total_count = len(similars_total_count)

	count_similar = 0

	for j in range(len(scores_list)):
		if scores_list[j][1] == 1:
			count += 1
			sum_prec += count/(j+1)
			last_index = j+1

			if flag == 0:
				sum_ranks += 1.0/(j+1)
				flag = 1

			if j == 0:
				top_1 += 1

			if j < 5:
				top_5 += 1

		else:
			if count_similar < similars_total_count:
				sum_prec += count/(j+1)


	if last_index > 0:
		sum_prec /= last_index

	sum_av_prec += sum_prec
	num_samples += 1

	return sum_av_prec, sum_ranks, num_samples, top_5, top_1



def createSamplesDataset(path, CROSS_DOMAIN):

	dataset = []

	if CROSS_DOMAIN == False:
		with open(path) as f:
			for line in f:
				split = line.split('\t')
				q = split[0]
				pos = split[1].split()
				rest_q = split[2].split()

				dataset.append([q, pos, rest_q])
	else:
		pass

	return dataset



def ComputeSimilarity(path, vectorizer, questions_dict, CROSS_DOMAIN):

	dataset = createSamplesDataset(path, CROSS_DOMAIN)

	sum_av_prec = 0.0
	sum_ranks = 0.0
	num_samples = 0.0
	top_5 = 0.0
	top_1 = 0.0

	for sample in dataset:
		q = sample[0]
		pos = sample[1]
		rest_q = sample[2]

		try:
			query_vector = vectorizer.transform([questions_dict[q]]).toarray()
		except:
			continue

		vector_label_list = createVectorLabelTuples(rest_q, questions_dict, q, pos, vectorizer)

		all_q_feature_vectors = [x[0] for x in vector_label_list]

		cs = cosine_similarity(all_q_feature_vectors, query_vector)
			
		cs_label_pair = []
		for index, question in enumerate(vector_label_list):
			cs_label_pair.append((cs[index], question[1]))

		scores_list = sorted(cs_label_pair, reverse = True, key=itemgetter(0))
		sum_av_prec, sum_ranks, num_samples, top_5, top_1 = updateScores(scores_list, sum_av_prec, sum_ranks, num_samples, top_5, top_1)

	_map = sum_av_prec/num_samples
	_mrr = sum_ranks/num_samples
	_pat5 = top_5/(num_samples*5)
	_pat1 = top_1/num_samples
	print('MAP: {:.3f}'.format(_map))
	print('MRR: {:.3f}'.format(_mrr))
	print('P@1: {:.3f}'.format(_pat1))
	print('P@5: {:.3f}'.format(_pat5))


stop_words_used = None #or get_stop_words('en')
NGRAM_RANGE = (1,3)

vocabulary = buildDictionary()
#vocabulary = None
#vocabulary could be None if we would like TfidfVectorizer to build a comprehensive vocabulary of the data (takes a very long time)

vectorizer, questions_dict = vectorizeData(PATH_QUESTION_CORPUS, False, vocabulary, stop_words_used, NGRAM_RANGE, 1.0, 1)
all_question_numbers = questions_dict.keys()
print "*******DEV**********"
ComputeSimilarity(PATH_DEV, vectorizer, questions_dict, False)
print "*******TEST**********"
ComputeSimilarity(PATH_TEST, vectorizer, questions_dict, False)
