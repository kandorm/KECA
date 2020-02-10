#-*-coding:utf-8 -*-
import sys
import numpy as np
import pickle
from qa.data_process import Vocab
np.random.seed(1337)

# python preprocess.py wikiqa
# python preprocess.py trecqa

class Util(object):
	@staticmethod
	def generate_vocab(file_list,output_file):
		vf = open(output_file,'w',encoding="utf-8")
		vocab = {}
		for filename in file_list:
			with open(filename, 'r', encoding="utf-8") as f:
				for line in f:
					sents = line.strip().split("\t")
					for i in range(2):
						words = sents[i].lower().split()
						for word in words:
							if word not in vocab:
								vocab[word] = len(vocab) + 1
		for word,index in sorted(vocab.items(),key=lambda x:x[1]):
				vf.write(str(index)+"\t"+word+"\n")

	@staticmethod
	def generate_embed(vocab_file,glovec_file,output_file):
		vo = Vocab(vocab_file,80000)
		embeding_list = [[] for i in range(vo.NumIds())]
		padding = np.random.randn(300) * 0.2 
		embeding_list[0] = padding
		count = 0
		with open(glovec_file, 'r', encoding="utf-8") as f:
			for line in f:
				units = line.strip().split(" ")
				word = units[0].lower()
				if word in vo._word_to_id:
					vector = list(map(float,units[1:]))
					index = vo.WordToId(word)
					if len(embeding_list[index]) == 0:
						embeding_list[index] = vector
					else:
						continue
					count += 1
		print("{} out of {} words have embedding".format(count+1, vo.NumIds()))
		for i in range(vo.NumIds()):
			if len(embeding_list[i]) == 0:
				temp_vec = (np.random.randn(300) * 0.2).tolist()
				embeding_list[i] = temp_vec
		embedding_vec = np.array(embeding_list)
		print("word embedding shape:{}".format(embedding_vec.shape))
		np.save(output_file, embedding_vec)
	
	@staticmethod
	def generate_ent_embed(infile_path, embedding_size, output_file):
		embeding_list = [[0] * embedding_size]
		with open(infile_path, "rb") as infile:
			for row in infile:
				items = row.strip().split()
				embeding_list.append(list(map(float,items[1:])))
		embedding_vec = np.array(embeding_list)
		print("entity embedding shape:{}".format(embedding_vec.shape))
		np.save(output_file, embedding_vec)

	@staticmethod
	def generate_data(input_file,vocab_file,label_file,output_file):
		vo = Vocab(vocab_file,80000)

		print("Loading entity data from "+label_file)
		vec_dict = {}
		with open(label_file, 'r', encoding="utf-8") as f:
			for line in f:
				l=line.lower().strip().split("\t")
				vec = [list(map(int,x.split(" "))) for x in l[1:]]
				vec_dict[l[0]]=vec

		ff = open(output_file,"wb")
		data = []
		que_set = set()
		with open(input_file, 'r', encoding="utf-8") as f:
			for line in f:
				units = line.lower().strip().split("\t")
				que_set.add(units[0])
				question = list(map(int,vo.Encode(units[0].split())))
				answer = list(map(int,vo.Encode(units[1].split())))
				label = int(units[2])
				que_ent = vec_dict.get(units[0], [[0]*5])
				ans_ent = vec_dict.get(units[1], [[0]*5])
				data.append((question,answer,label,que_ent,ans_ent))
		print("Question num:{}".format(len(que_set)))
		pickle.dump(data,ff)

	@staticmethod
	def generate_answer(file_list, output_file):
		answer_list = []
		for filename in file_list:
			data = pickle.load(open(filename,'rb'))
			for item in data:
				if item[2] == 0:
					answer_list.append((item[1],item[4]))
		pickle.dump(answer_list,open(output_file,'wb'))


if __name__ == "__main__":
	task = sys.argv[1]
	if task == "wikiqa":
		print("generate vocab")
		Util.generate_vocab(file_list=["./data/raw_data/WikiQA/wikiQA-train.txt","./data/raw_data/WikiQA/wikiQA-dev.txt","./data/raw_data/WikiQA/wikiQA-test.txt"],output_file="./data/wikiqa/vocab_wiki.txt")
		print("generate emb")
		Util.generate_embed(vocab_file="./data/wikiqa/vocab_wiki.txt",glovec_file="./data/glove/glove.840B.300d.txt",output_file="./data/wikiqa/wikiqa_glovec.npy")
		print("generate entity emb")
		Util.generate_ent_embed(infile_path="./data/fb5m/fb5m-wiki.transE", embedding_size=64, output_file="./data/wikiqa/wikiqa_fb5m_ent.npy")
		print("generate data pkl")
		Util.generate_data("./data/raw_data/WikiQA/wikiQA-train.txt","./data/wikiqa/vocab_wiki.txt","./data/raw_data/WikiQA/wikiQA-train.txt.labeled","./data/wikiqa/wiki_train.pkl")
		Util.generate_data("./data/raw_data/WikiQA/wikiQA-dev.txt","./data/wikiqa/vocab_wiki.txt","./data/raw_data/WikiQA/wikiQA-dev.txt.labeled","./data/wikiqa/wiki_dev.pkl")
		Util.generate_data("./data/raw_data/WikiQA/wikiQA-test.txt","./data/wikiqa/vocab_wiki.txt","./data/raw_data/WikiQA/wikiQA-test.txt.labeled","./data/wikiqa/wiki_test.pkl")
		Util.generate_answer(["./data/wikiqa/wiki_train.pkl"], "./data/wikiqa/wiki_answer_train.pkl")  # random answer from train data for batch training
	elif task == "trecqa":
		print("generate vocab")
		Util.generate_vocab(file_list=["./data/raw_data/TrecQA/trecQA-all-train.txt","./data/raw_data/TrecQA/trecQA-clean-dev.txt","./data/raw_data/TrecQA/trecQA-clean-test.txt"], output_file="./data/trecqa/vocab_trec.txt")
		print("generate emb")
		Util.generate_embed(vocab_file="./data/trecqa/vocab_trec.txt", glovec_file="./data/glove/glove.840B.300d.txt", output_file="./data/trecqa/trecqa_glovec.npy")
		print("generate entity emb")
		Util.generate_ent_embed(infile_path="./data/fb5m/fb5m-trec.transE", embedding_size=64, output_file="./data/trecqa/trecqa_fb5m_ent.npy")
		print("generate data pkl")
		Util.generate_data("./data/raw_data/TrecQA/trecQA-all-train.txt","./data/trecqa/vocab_trec.txt","./data/raw_data/TrecQA/trecQA-all-train.txt.labeled","./data/trecqa/trec_train.pkl")
		Util.generate_data("./data/raw_data/TrecQA/trecQA-clean-dev.txt","./data/trecqa/vocab_trec.txt","./data/raw_data/TrecQA/trecQA-clean-dev.txt.labeled","./data/trecqa/trec_dev.pkl")
		Util.generate_data("./data/raw_data/TrecQA/trecQA-clean-test.txt","./data/trecqa/vocab_trec.txt","./data/raw_data/TrecQA/trecQA-clean-test.txt.labeled","./data/trecqa/trec_test.pkl")
		Util.generate_answer(["./data/trecqa/trec_train.pkl"], "./data/trecqa/trec_answer_train.pkl")  # random answer from train data for batch training
	else:
		sys.stderr.write("illegal param")
