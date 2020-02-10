#-*-coding:utf-8 -*-
import sys
import numpy as np
import random
from collections import namedtuple
import pickle

np.random.seed(1337)
random.seed(1337)

ModelParam = namedtuple("ModelParam","hidden_dim,enc_timesteps,dec_timesteps,batch_size,random_size,k_value_que,k_value_ans,lr,n_entity,entity_dim")

UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
class Vocab(object):
	def __init__(self, vocab_file, max_size):
		self._word_to_id = {}
		self._id_to_word = {}
		self._count = 0
		before_list = [PAD_TOKEN]
		for word in before_list:
			self.CreateWord(word)
		with open(vocab_file, 'r', encoding="utf-8") as vocab_f:
			for line in vocab_f:
				pieces = line.strip().split("\t")
				if len(pieces) != 2:
					sys.stderr.write('Bad line: %s\n' % line)
					continue
				if pieces[1] in self._word_to_id:
					raise ValueError('Duplicated word: %s.' % pieces[1])
				self._word_to_id[pieces[1]] = int(pieces[0])
				self._id_to_word[int(pieces[0])] = pieces[1]
				self._count += 1
				if self._count > max_size-1:
					sys.stderr.write('Too many words: >%d.' % max_size)
					break

	def WordToId(self, word):
		if word not in self._word_to_id:
			return self._count  # len(self._word_to_id) for '<UNK>'
		return self._word_to_id[word]

	def IdToWord(self, word_id):
		if word_id not in self._id_to_word:
			raise ValueError('id not found in vocab: %d.' % word_id)
		return self._id_to_word[word_id]

	def NumIds(self):
		return self._count

	def CreateWord(self,word):
		if word not in self._word_to_id:
			self._word_to_id[word] = self._count
			self._id_to_word[self._count] = word
			self._count += 1

	def Revert(self,indices):
		vocab = self._id_to_word
		return [vocab.get(i, UNKNOWN_TOKEN) for i in indices]

	def Encode(self,indices):
		vocab = self._word_to_id
		return [vocab.get(i, self._count) for i in indices]


class DataGenerator(object):
	# Dataset class
	def __init__(self, vocab, model_param, answer_file=""):
		self.vocab = vocab
		self.param = model_param
		if answer_file != "":
			self.answers = pickle.load(open(answer_file,'rb'))

	def padq(self, data):
	    return self.pad(data, self.param.enc_timesteps)

	def pada(self, data):
	    return self.pad(data, self.param.dec_timesteps)

	def pad(self, data, maxlen=None):
	    from keras.preprocessing.sequence import pad_sequences
	    return pad_sequences(data, maxlen=maxlen, padding='post', truncating='post', value=0)

	def entpadq(self, data):
		m = data[:self.param.enc_timesteps]
		m.extend([[0]*self.param.n_entity]*(self.param.enc_timesteps-len(m)))
		return np.array(m)

	def entpada_all(self, data):
		m = data[:]
		for i, d in enumerate(m):
			m[i] = self.entpada(d)
		return np.array(m)

	def entpada(self, data):
		m = data[:self.param.dec_timesteps]
		m.extend([[0]*self.param.n_entity]*(self.param.dec_timesteps-len(m)))
		return np.array(m)

	def wikiQaGenerate(self,filename,flag="basic"):
		print("reading from {}".format(filename))

		data = pickle.load(open(filename,'rb'))
		question_dic = {}
		question = list()
		answer = list()
		label = list()
		question_len = list()
		answer_len = list()
		question_ent = list()
		answer_ent = list()
		
		answer_size = list()
		
		for item in data:
			question_dic.setdefault(str(item[0]),{})
			question_dic[str(item[0])].setdefault("question",[])
			question_dic[str(item[0])].setdefault("answer",[])
			question_dic[str(item[0])].setdefault("label",[])
			question_dic[str(item[0])].setdefault("que_ent",[])
			question_dic[str(item[0])].setdefault("ans_ent",[])
			question_dic[str(item[0])]["question"].append(item[0])
			question_dic[str(item[0])]["answer"].append(item[1])
			question_dic[str(item[0])]["label"].append(item[2])
			question_dic[str(item[0])]["que_ent"].append(item[3])
			question_dic[str(item[0])]["ans_ent"].append(item[4])

		delCount = 0
		for key in question_dic.keys():
			if sum(question_dic[key]["label"]) == 0:
				delCount += 1
				del(question_dic[key])
		print("delCount:{}".format(delCount))

		for item in question_dic.values():
			good_answer = []
			bad_answer = []
			for i in range(len(item["question"])):
				if item["label"][i] == 1:
					good_answer.append((item["answer"][i], item["ans_ent"][i]))
				else:
					bad_answer.append((item["answer"][i], item["ans_ent"][i]))
			good_length = len(good_answer)
			trash_sample = self.param.random_size
			if len(item["answer"]) >= self.param.random_size:
				if good_length >= self.param.random_size:
					temp_answer = random.sample(good_answer, self.param.random_size)
					temp_answer, temp_ans_ent = zip(*temp_answer)
					temp_answer = list(temp_answer)
					temp_ans_ent = list(temp_ans_ent)
					temp_label = [1 / float(self.param.random_size)] * self.param.random_size
				else:
					temp_answer = good_answer[:]
					temp_answer.extend(random.sample(bad_answer, self.param.random_size - good_length))
					temp_answer, temp_ans_ent = zip(*temp_answer)
					temp_answer = list(temp_answer)
					temp_ans_ent = list(temp_ans_ent)
					temp_label = [1 / float(good_length)] * good_length
					temp_label.extend([0.0] * (self.param.random_size - good_length))
			else:
				temp_answer = good_answer + bad_answer
				temp_answer.extend(random.sample(self.answers, self.param.random_size-len(item["answer"])))
				temp_answer, temp_ans_ent = zip(*temp_answer)
				temp_answer = list(temp_answer)
				temp_ans_ent = list(temp_ans_ent)
				temp_label = [1 / float(good_length)] * good_length
				temp_label.extend([0.0] * (self.param.random_size - good_length))
				trash_sample = len(item["question"])

			question.append(self.padq([item["question"][0]])[0])
			answer.append(self.pada(temp_answer))
			label.append(temp_label)

			question_ent.append(self.entpadq(item["que_ent"][0]))
			answer_ent.append(self.entpada_all(temp_ans_ent))

			ans_length = [[1]*len(single_ans) for single_ans in temp_answer]
			answer_len.append(self.pada(ans_length))

			que_length = [1]*len(item["question"][0])
			question_len.append(self.padq([que_length])[0])

			answer_size.append([1]*trash_sample + [0]*(self.param.random_size-trash_sample))

		question = np.array(question)
		answer = np.array(answer)
		label = np.array(label)
		question_len = np.array(question_len)
		answer_len = np.array(answer_len)
		question_ent = np.array(question_ent)
		answer_ent = np.array(answer_ent)
		answer_size = np.array(answer_size)

		print("question shape:{}".format(question.shape))
		print("answer shape:{}".format(answer.shape))
		print("label shape:{}".format(label.shape))
		print("question length shape:{}".format(question_len.shape))
		print("answer length shape:{}".format(answer_len.shape))
		print("question_ent shape:{}".format(question_ent.shape))
		print("answer_ent shape:{}".format(answer_ent.shape))

		if flag == "size":
			return question, answer, label, question_len, answer_len, question_ent, answer_ent, answer_size
		return question, answer, label, question_len, answer_len, question_ent, answer_ent

	def trecQaGenerate(self,filename,flag="basic"):
		print("reading from {}".format(filename))

		data = pickle.load(open(filename,'rb'))
		question_dic = {}
		question = list()
		answer = list()
		label = list()
		question_len = list()
		answer_len = list()
		question_ent = list()
		answer_ent = list()
		answer_size = list()
		for item in data:
			question_dic.setdefault(str(item[0]),{})
			question_dic[str(item[0])].setdefault("question",[])
			question_dic[str(item[0])].setdefault("answer",[])
			question_dic[str(item[0])].setdefault("label",[])
			question_dic[str(item[0])].setdefault("que_ent",[])
			question_dic[str(item[0])].setdefault("ans_ent",[])
			question_dic[str(item[0])]["question"].append(item[0])
			question_dic[str(item[0])]["answer"].append(item[1])
			question_dic[str(item[0])]["label"].append(item[2])
			question_dic[str(item[0])]["que_ent"].append(item[3])
			question_dic[str(item[0])]["ans_ent"].append(item[4])

		delCount = 0
		for key in question_dic.keys():
			if sum(question_dic[key]["label"]) == 0:
				delCount += 1
				del(question_dic[key])
		print("delCount:{}".format(delCount))

		for item in question_dic.values():
			good_answer = []
			bad_answer = []
			for i in range(len(item["question"])):
				if item["label"][i] == 1:
					good_answer.append((item["answer"][i], item["ans_ent"][i]))
				else:
					bad_answer.append((item["answer"][i], item["ans_ent"][i]))
			good_length = len(good_answer)

			if good_length >= self.param.random_size // 2:
				good_answer = random.sample(good_answer, self.param.random_size//2)
				good_length = len(good_answer)

			trash_sample = self.param.random_size
			if len(bad_answer) >= self.param.random_size - good_length:
				temp_answer = good_answer[:]
				temp_answer.extend(random.sample(bad_answer, self.param.random_size - good_length))
				temp_answer, temp_ans_ent = zip(*temp_answer)
				temp_answer = list(temp_answer)
				temp_ans_ent = list(temp_ans_ent)
				temp_label = [1 / float(good_length)] * good_length
				temp_label.extend([0.0] * (self.param.random_size - good_length))
			else:
				temp_answer = good_answer + bad_answer
				trash_sample = len(temp_answer)
				temp_answer.extend(random.sample(self.answers, self.param.random_size-len(temp_answer)))
				temp_answer, temp_ans_ent = zip(*temp_answer)
				temp_answer = list(temp_answer)
				temp_ans_ent = list(temp_ans_ent)
				temp_label = [1 / float(good_length)] * good_length
				temp_label.extend([0.0] * (self.param.random_size - good_length))

			question.append(self.padq([item["question"][0]])[0])
			answer.append(self.pada(temp_answer))
			label.append(temp_label)

			question_ent.append(self.entpadq(item["que_ent"][0]))
			answer_ent.append(self.entpada_all(temp_ans_ent))

			ans_length = [[1]*len(single_ans) for single_ans in temp_answer]
			answer_len.append(self.pada(ans_length))

			que_length = [1]*len(item["question"][0])
			question_len.append(self.padq([que_length])[0])
			
			answer_size.append([1]*trash_sample + [0]*(self.param.random_size-trash_sample))

		question = np.array(question)
		answer = np.array(answer)
		label = np.array(label)
		question_len = np.array(question_len)
		answer_len = np.array(answer_len)
		question_ent = np.array(question_ent)
		answer_ent = np.array(answer_ent)
		answer_size = np.array(answer_size)

		print("question shape:{}".format(question.shape))
		print("answer shape:{}".format(answer.shape))
		print("label shape:{}".format(label.shape))
		print("question length shape:{}".format(question_len.shape))
		print("answer length shape:{}".format(answer_len.shape))
		print("question_ent shape:{}".format(question_ent.shape))
		print("answer_ent shape:{}".format(answer_ent.shape))

		if flag == "size":
			return question, answer, label, question_len, answer_len, question_ent, answer_ent, answer_size
		return question, answer, label, question_len, answer_len, question_ent, answer_ent

	def EvaluateGenerate(self, filename):
		print("load data from {}".format(filename))
		data = pickle.load(open(filename, 'rb'))
		question_dic = {}
		for item in data:
			question_dic.setdefault(str(item[0]),{})
			question_dic[str(item[0])].setdefault("question",[])
			question_dic[str(item[0])].setdefault("answer",[])
			question_dic[str(item[0])].setdefault("label",[])
			question_dic[str(item[0])].setdefault("que_ent",[])
			question_dic[str(item[0])].setdefault("ans_ent",[])
			question_dic[str(item[0])]["question"].append(item[0])
			question_dic[str(item[0])]["answer"].append(item[1])
			question_dic[str(item[0])]["label"].append(item[2])
			question_dic[str(item[0])]["que_ent"].append(item[3])
			question_dic[str(item[0])]["ans_ent"].append(item[4])

		delCount = 0
		for key in question_dic.keys():
			question_dic[key]["ques_len"] = self.padq([[1]*len(single_que) for single_que in question_dic[key]["question"]])
			question_dic[key]["ans_len"] = self.pada([[1]*len(single_ans) for single_ans in question_dic[key]["answer"]])
			question_dic[key]["question"] = self.padq(question_dic[key]["question"])
			question_dic[key]["answer"] = self.pada(question_dic[key]["answer"])
			
			question_dic[key]["que_ent"] = np.array([self.entpadq(question_dic[key]["que_ent"][0])] * len(question_dic[key]["que_ent"]))
			question_dic[key]["ans_ent"] = self.entpada_all(question_dic[key]["ans_ent"])
			
			if sum(question_dic[key]["label"]) == 0:
				delCount += 1
				del(question_dic[key])
		print("del count:{}".format(delCount))
		print("question count:{}".format(len(question_dic)))
		return question_dic
