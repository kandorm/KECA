#-*-coding:utf-8-*-
import numpy as np
import logging
import sys
import os
import h5py
import argparse
from qa.model_factory import ModelFactory
from qa.data_process import Vocab, DataGenerator, ModelParam

np.random.seed(1337)

log = logging.getLogger("output")
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='log.txt',
                filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

parser = argparse.ArgumentParser(description='Knowledge Enhanced Latent Relevance Mining For Question Answering')
parser.add_argument('-t','--task',type=str,default="wikiqa",help='task: trecqa or wikiqa')
parser.add_argument('-m',"--model",type=str,default="listwise",help='model: listwise, k_max or k_threhold')
parser.add_argument('-d',"--hidden_dim",type=int,default=300,help='dimension of the hidden layer')
parser.add_argument('-e',"--patience",type=int,default=10,help='patience')
parser.add_argument('-l',"--lr",type=float,default=0.001,help='learning rate')
parser.add_argument('-k_q',"--k_value_que",type=float,default=5,help='k value in k_max or k_threshold for question attention')
parser.add_argument('-k_a',"--k_value_ans",type=float,default=5,help='k value in k_max or k_threshold for answer attention')
parser.add_argument('-b',"--batch_size",type=int,default=4,help='batch_size')
parser.add_argument('-p',"--pre_train",type=int,default=10,help='pre_train')
parser.add_argument('--no_entity_cmp', action='store_false', help='do not use entity compare layer', dest='entity_compare')
parser.add_argument('--no_cxt_weight_mixed', action='store_false', help='do not mix context weight ', dest='cxt_weight_mixed')
parser.add_argument('--no_ent_weight_mixed', action='store_false', help='do not mix entity weight ', dest='ent_weight_mixed')


def result_log(str):
    ff = open("./result_note.txt",'a')
    ff.write(str+"\n")
    ff.close()


def specific_load_epoch(model,epoch,prefix):
    assert os.path.exists('model/'+prefix+'_weights_epoch_%s.h5' %
                          epoch), 'Weights at epoch %s not found' % epoch
    filename = 'model/'+prefix+'_weights_epoch_%s.h5' % epoch
    h5_file = h5py.File(filename,'r')
    weight = []
    for i in range(len(h5_file.keys())):
        weight.append(h5_file['weight'+str(i)][:])
    model.set_weights(weight)


def specific_save_epoch(model,epoch,prefix,global_mark_copy):
    if not os.path.exists('model/'):
        os.makedirs('model/')
    filename = 'model/'+prefix+'_weights_epoch_%s.h5' % (str(epoch)+"_"+global_mark_copy)
    h5_file = h5py.File(filename,'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        h5_file.create_dataset('weight'+str(i),data=weight[i])
    h5_file.close()


def prog_bar(so_far, total, n_bars=20):
    n_complete = int(so_far * n_bars / total)
    if n_complete >= n_bars - 1:
        sys.stderr.write('\r[' + '=' * n_bars + ']')
    else:
        s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * \
            (n_bars - n_complete) + ']'
        sys.stderr.write(s)


def task_data_ready(task,model_param):
    if task == "wikiqa":
        vocab_all = Vocab("./data/wikiqa/vocab_wiki.txt", max_size=80000)
        data_generator = DataGenerator(vocab_all, model_param,"./data/wikiqa/wiki_answer_train.pkl")
        embedding_file = "./data/wikiqa/wikiqa_glovec.npy"
        ent_embedding_file = "./data/wikiqa/wikiqa_fb5m_ent.npy"
        dev_data = data_generator.EvaluateGenerate("./data/wikiqa/wiki_dev.pkl")
        test_data = data_generator.EvaluateGenerate("./data//wikiqa/wiki_test.pkl")    
    elif task == "trecqa":
        vocab_all = Vocab("./data/trecqa/vocab_trec.txt", max_size=80000)
        data_generator = DataGenerator(vocab_all, model_param,"./data/trecqa/trec_answer_train.pkl")
        embedding_file = "./data/trecqa/trecqa_glovec.npy"
        ent_embedding_file = "./data/trecqa/trecqa_fb5m_ent.npy"
        dev_data = data_generator.EvaluateGenerate("./data/trecqa/trec_dev.pkl")
        test_data = data_generator.EvaluateGenerate("./data/trecqa/trec_test.pkl")
    return vocab_all, data_generator, embedding_file, ent_embedding_file, dev_data, test_data


def main(args):
    global_mark = args.task + "_" + args.model
    print(str(args.pre_train)+" model")
    if args.task == "wikiqa":
        model_param = ModelParam(hidden_dim=args.hidden_dim, enc_timesteps=25, dec_timesteps=90, batch_size=args.batch_size, random_size=15, lr=args.lr, k_value_que=args.k_value_que,k_value_ans=args.k_value_ans, n_entity=5, entity_dim=64)
    elif args.task == "trecqa":
        model_param = ModelParam(hidden_dim=args.hidden_dim, enc_timesteps=30, dec_timesteps=70, batch_size=args.batch_size, random_size=40, lr=args.lr, k_value_que=args.k_value_que,k_value_ans=args.k_value_ans, n_entity=5, entity_dim=64)

    logging.info(model_param.__str__())

    vocab_all, data_generator, embedding_file, ent_embedding_file, dev_data, test_data = task_data_ready(args.task,model_param)
    train_model, predict_model = ModelFactory.get_model(model_param, embedding_file, ent_embedding_file, model_type=args.model, cxt_weight_mixed=args.cxt_weight_mixed, ent_weight_mixed=args.ent_weight_mixed, entity_compare=args.entity_compare)

    def data_evaluate(epoch,small_evaluate_data,flag):
        c_1_j = 0        
        c_2_j = 0
        for i, d in enumerate(small_evaluate_data.values()):
            prog_bar(i, len(small_evaluate_data))
            question = d["question"]
            answers = d["answer"]
            question_len = d["ques_len"]
            ans_len = d["ans_len"]
            que_ent = d["que_ent"]
            ans_ent = d["ans_ent"]
            sims = predict_model.predict([question,answers,question_len,ans_len,que_ent,ans_ent],batch_size=len(question))
            sims = sims[:,0]
            rank_index = np.argsort(sims).tolist()[::-1]
            score = 0.0
            count = 0.0
            for i in range(1,len(sims)+1):
                if d["label"][rank_index[i-1]] == 1:
                    count += 1
                    score += count / i
            for i in range(1,len(sims)+1):
                if d["label"][rank_index[i-1]] == 1:
                    c_2_j += 1/float(i)
                    break
            c_1_j += score / count

        MAP = c_1_j / float(len(small_evaluate_data))
        MRR = c_2_j / float(len(small_evaluate_data))
        print("")
        logging.info(global_mark + " evaluate on "+ flag +" data at epoch "+str(epoch)+' MAP: %f' % MAP)
        logging.info(" evaluate on "+ flag + " data at epoch "+str(epoch)+' MRR: %f' % MRR)
        result_log(global_mark+" evaluate on "+ flag + " data at epoch "+str(epoch)+' MAP: %f' % MAP)
        result_log(global_mark+" evaluate on "+ flag + " data at epoch "+str(epoch)+' MRR: %f' % MRR)
        return MAP,MRR

    if args.model != "listwise":
        reload_epoch = args.pre_train
        assert os.path.exists('model/train_weights_epoch_%s.h5' % (str(reload_epoch)+"_"+args.task+"_listwise")), "please pre-train listwise approach"
        specific_load_epoch(train_model, str(reload_epoch)+"_"+args.task+"_listwise", "train")

    best_epoch = 0
    best_map = 0
    score_list = []
    epoch = 0
    epoch_not_improved = 0
    early_stop = False
    test_best_score = [0, 0]
    while True:
        if early_stop:
            print("Early Stopping.")
            break

        epoch += 1
        if args.task == "wikiqa":
            train_filename = "./data/wikiqa/wiki_train.pkl"        
            questions, answers, label, question_len, answer_len, question_ent, answer_ent = data_generator.wikiQaGenerate(train_filename, "basic")
        elif args.task == "trecqa":
            train_filename = "./data/trecqa/trec_train.pkl"        
            questions, answers, label, question_len, answer_len, question_ent, answer_ent = data_generator.trecQaGenerate(train_filename, "basic")
        logging.info('Fitting epoch %d' % epoch)


        train_model.fit([questions, answers,question_len,answer_len,question_ent,answer_ent], label,epochs=1, batch_size=model_param.batch_size, validation_split=0, verbose=1,shuffle=True)           
        specific_save_epoch(train_model,epoch, prefix="train",global_mark_copy = global_mark)
        specific_save_epoch(predict_model,epoch, prefix="predict",global_mark_copy = global_mark)

        dev_map,dev_mrr = data_evaluate(epoch,dev_data,"dev")
        test_map,test_mrr = data_evaluate(epoch,test_data,"test")
        if dev_map > best_map:
            epoch_not_improved = 0
            best_map = dev_map
            best_epoch = epoch
            score_list = [dev_map,dev_mrr,test_map,test_mrr]
        else:
            epoch_not_improved += 1
            if epoch_not_improved > args.patience:
                early_stop = True

        if test_map > test_best_score[0]:
            test_best_score[0] = test_map
            test_best_score[1] = test_mrr
            epoch_not_improved = 0

    logging.info("best model at epoch "+str(best_epoch))
    logging.info("the dev score of best model: MAP_"+str(score_list[0])+" MRR_"+str(score_list[1]))
    logging.info("the test score of best model: MAP_"+str(score_list[2])+" MRR_"+str(score_list[3]))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
