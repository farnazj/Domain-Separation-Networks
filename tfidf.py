from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
from operator import itemgetter
from sklearn import metrics
import train.meter as meter
import numpy as np
import gzip
import string
import random


PATH_EMB = "./askubuntu/vector/vectors_pruned.200.txt.gz"
PATH_SOURCE_QUESTION_CORPUS = "./askubuntu/text_tokenized.txt.gz"
PATH_TARGET_QUESTION_CORPUS =  "./Android/corpus.tsv.gz"

PATHS_DEV = ["./Android/dev.pos.txt", "./Android/dev.neg.txt"]  #["./askubuntu/dev.txt"] #
PATHS_TEST = ["./Android/test.pos.txt", "./Android/test.neg.txt"] #["./askubuntu/test.txt"]  #

TEXTMAX_LENGTH = 100 #int or None
QCOUNT = 20
CROSS_DOMAIN = True



def readQCorpus(filename):
	'''
	readQCorpus reads the corpus file and returns a dictionary of question ids with their corresponding (trimmed) text
	'''

	data = {}
	with gzip.open(filename) as gfile:
		for row in gfile:
			row_arr = row.split()

			row_string = string.join(row_arr[1:TEXTMAX_LENGTH + 1])

			data[row_arr[0]] = row_string

	return data



def vectorizeData(corpus_data, binary_count, vocab, stop_words_, ngram_range_, max_df_, min_df_):

	vectorizer = TfidfVectorizer(input ='content', binary=binary_count, vocabulary=vocab, stop_words= stop_words_, ngram_range=ngram_range_, max_df=max_df_, min_df= min_df_)

	vectorizer.fit(corpus_data.values())

	return vectorizer


def buildDictionary():
	'''
	Builds the dictionary from the embedding file, to be used by tfidfvectorizer.
	'''

	vocabulary = []
	with gzip.open(PATH_EMB) as gfile:
		for line in gfile:
			word = line.split()[0]
			vocabulary.append(word)

	return vocabulary


def createVectorLabelTuples(all_q, questions_dict, q, list_of_pos, vectorizer):
	'''
	Iterates over the questions of a sample and appends a binary label to each, corresponding to whether
	the question is similar or dissimilar to the query question
	'''
	vector_label_list = []

	for item in all_q: #item is a number referring to a question

		vector = vectorizer.transform([questions_dict[item]]).toarray()
		#vector[0]?
		label = 1 if item in list_of_pos else 0
		vector_label_list.append((vector[0], label))

	return vector_label_list



def updateScores(scores_list, sum_av_prec, sum_ranks, num_samples, top_5, top_1):

	count = 0.0
	last_index = -1
	sum_prec = 0.0
	flag = 0

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


	if last_index > 0:
		sum_prec /= count

	sum_av_prec += sum_prec
	num_samples += 1

	return sum_av_prec, sum_ranks, num_samples, top_5, top_1



def sanitizeSample(q, pos, negs, all_question_numbers):
	'''
	sanitizeSample ensures:
	* All questions per sample exist in the corpus
	* The number of questions per sample is exactly QCOUNT
	'''

	pos_sanitized = []
	all_q_sanitized = []

	all_pos_neg = pos + negs

	'''sanitizing questions; considering only those positive questions that exist in the question corpus. for negative questions that do not exist
	in the corpus, we replace them with a randomly drawn question from the list of all questions. If at the end, the mingled list of all positive
	and negative questions doesn't contain as many questions as MAX_QCOUNT, we randomly draw negative questions from the dataset.'''

	neg_candidates_needed = 0

	if q not in all_question_numbers: #if the query question does not exist in the corpus
		return None

	for question in all_pos_neg:
		if question in all_question_numbers:
			if question in pos:
				pos_sanitized.append(question)

			all_q_sanitized.append(question)

	if len(pos_sanitized) < 1: #if after taking out the pos questions that do not exist in the corpus, the list of pos questions is empty
		return None

	if len(all_q_sanitized) < QCOUNT:
		new_neg_candidates = []

		while len(all_q_sanitized) < QCOUNT:
			while True:
				neg_candidate = random.choice(all_question_numbers)
				if neg_candidate not in set().union(rest_q, pos, [q], new_neg_candidates):
					if neg_candidate in all_question_numbers:
						new_neg_candidates.append(neg_candidate)
						break

		all_q_sanitized.extend(new_neg_candidates)

	elif len(all_q_sanitized) > QCOUNT: #for the target dataset that has more than QCOUNT questions per sample

		extra = len(all_q_sanitized) - QCOUNT
		random.shuffle(all_q_sanitized)

		while extra:
			throw_away_index = random.randint(0, len(all_q_sanitized) - 1)
			if all_q_sanitized[throw_away_index] not in pos_sanitized:
				del all_q_sanitized[throw_away_index]
				extra -= 1

	random.shuffle(all_q_sanitized)

	return [pos_sanitized, all_q_sanitized]



def createSamplesDataset(path, CROSS_DOMAIN, questions_numbers):

	'''
	Returns a dictionary of question ids with values of the form: [list of similar questions, list of same similar questions and other (nonsimilar) questions]
	'''

	dataset = {}

	if CROSS_DOMAIN == False:
		with open(path[0]) as f:
			for line in f:
				split = line.split('\t')
				q = split[0]
				pos = split[1].split()
				rest_q = split[2].split()

				sanitized_qs = sanitizeSample(q, pos, list(set(rest_q) - set(pos)), questions_numbers)

				if sanitized_qs is not None:
					dataset[q] = sanitized_qs

	else: #pos and neg files from target domain
		pos_path, neg_path = path[0], path[1]

		#target_question_numbers = [] #target question ids that exists in the corpus
		target_dataset_all = {} #dictionary of question ids with values of the form: []

		with open(pos_path) as f:
			for line in f:
				q, pos = line.split()
				pos = pos.split()
				target_dataset_all[q] = [pos, []]


		with open(neg_path) as f:
			for line in f:
				q, neg = line.split()
				if q in target_dataset_all: #otherwise we ignore it, since there is not similar question for it
					target_dataset_all[q][1].append(neg)


		for q in target_dataset_all:
			sanitized_qs = sanitizeSample(q, target_dataset_all[q][0], target_dataset_all[q][1], questions_numbers)

			if sanitized_qs is not None:
				dataset[q] = sanitized_qs


	return dataset



def ComputeSimilarity(path, vectorizer, questions_dict, CROSS_DOMAIN):

	'''
	Expects a list of paths (containing a single element if validation data is from the same domain, two elements if the data is from another domain)
	'''

	all_q_scores = []

	questions_numbers = questions_dict.keys()

	dataset = createSamplesDataset(path, CROSS_DOMAIN, questions_numbers)

	sum_av_prec = 0.0
	sum_ranks = 0.0
	num_samples = 0.0
	top_5 = 0.0
	top_1 = 0.0
	auc_met = meter.AUCMeter()

	for q in dataset:

		pos = dataset[q][0]
		rest_q = dataset[q][1]

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

			auc_met.add(cs[index], question[1])

		if not CROSS_DOMAIN:
			scores_list = sorted(cs_label_pair, reverse = True, key=itemgetter(0))
			sum_av_prec, sum_ranks, num_samples, top_5, top_1 = updateScores(scores_list, sum_av_prec, sum_ranks, num_samples, top_5, top_1)
		else:
			all_q_scores.extend(cs_label_pair)



	if not CROSS_DOMAIN:
		_map = sum_av_prec/num_samples
		_mrr = sum_ranks/num_samples
		_pat5 = top_5/(num_samples*5)
		_pat1 = top_1/num_samples
		print 'MAP: {:.3f}'.format(_map)
		print 'MRR: {:.3f}'.format(_mrr)
		print 'P@1: {:.3f}'.format(_pat1)
		print 'P@5: {:.3f}'.format(_pat5)
	else:
		print 'AUC: {:.3f}'.format(auc_met.value(0.05))


stop_words_used = None #or get_stop_words('en')
NGRAM_RANGE = (1,3)

vocabulary = buildDictionary()
#vocabulary = None
#vocabulary could be None if we would like TfidfVectorizer to build a comprehensive vocabulary of the data (takes a very long time)

source_questions_dict = readQCorpus(PATH_SOURCE_QUESTION_CORPUS)

if CROSS_DOMAIN:
	questions_dict = readQCorpus(PATH_TARGET_QUESTION_CORPUS)
else:
	questions_dict = source_questions_dict

BINARY_COUNT = False


for stop_words_used in [None, get_stop_words('en')]:
	stop_word_status = "using stop words" if stop_words_used != None else "not using stop words"

	for NGRAM_RANGE in [(1,1), (1,2), (1,3)]:
		print NGRAM_RANGE

		for BINARY_COUNT in [True, False]:
			print BINARY_COUNT


			vectorizer = vectorizeData(source_questions_dict, BINARY_COUNT, vocabulary, stop_words_used, NGRAM_RANGE, 1.0, 1)

			print "*******DEV**********"
			ComputeSimilarity(PATHS_DEV, vectorizer, questions_dict, CROSS_DOMAIN)
			print "*******TEST**********"
			ComputeSimilarity(PATHS_TEST, vectorizer, questions_dict, CROSS_DOMAIN)

			print "\n"
