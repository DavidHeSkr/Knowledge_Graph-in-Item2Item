#coding:utf-8
import numpy as np
import pandas as pd
#import tensorflow as tf
import tensorflow as tf
import os
import time
import datetime
import ctypes
import json
import random
#tf.enable_eager_execution()

class Config(object):
	'''
	use ctypes to call C functions from python and set essential parameters.
	'''
	def __init__(self):
		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../release/Base.so'))
		self.test_flag = False
		self.in_path = None
		self.out_path = None
		self.bern = 0
		self.hidden_size = 100
		self.ent_size = self.hidden_size
		self.rel_size = self.hidden_size
		self.train_times = 0
		self.margin = 1.0
		self.nbatches = 100
		self.negative_ent = 1
		self.negative_rel = 0
		self.workThreads = 1
		self.alpha = 0.001
		self.lmbda = 0.000
		self.log_on = 1
		self.exportName = None
		self.importName = None
		self.export_steps = 0
		self.opt_method = "SGD"
		self.optimizer = None
		self.test_link_prediction = False
		self.test_triple_classification = False
		self.early_stopping = None # It expects a tuple of the following: (patience, min_delta)

		self.rel2movie = None
		self.rel2ids = None
		self.movie2rels = None
		self.entTotal = None
		self.relTotal = None

		self.sample_node_neighbour = None
		self.sample_node_number = None

		self.max_rel = 20
		self.rel_type = 2

	# prepare for train and test
	def init(self):
		self.trainModel = None
		#if self.in_path != None:
		self.read_files_()
		self.batch_size = self.sample_node_number * self.sample_node_neighbour * 2
		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
		#print("self.batch_size", self.batch_size)
		#print("self.batch_seq_size", self.batch_seq_size)
		self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		self.batch_type_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
		if self.test_link_prediction:
			self.init_link_prediction()
		if self.test_triple_classification:
			self.init_triple_classification()

	def read_files_(self):
		with open('rel2movies.json') as jsonfile:
			self.rel2movie = json.load(jsonfile)
		with open('rel2ids.json') as jsonfile:
			self.rel2ids = json.load(jsonfile)
		self.movie2rels = pd.read_csv("movie2rels.csv")

		with open('new_u2m.json') as jsonfile:
			rel2movie = json.load(jsonfile)
		rel2movie_ = {}
		for i in rel2movie:
			if i not in rel2movie_:
				rel2movie_[int(i)] = []
			for j in rel2movie[i]:
				rel2movie_[int(i)].append(int(j))
		self.u2m = rel2movie_

		with open('new_m2u.json') as jsonfile:
			movie2rel = json.load(jsonfile)
		movie2rel_ = {}
		for m in movie2rel:
			if m not in movie2rel_:
				movie2rel_[int(m)] = []
			for n in movie2rel[m]:
				movie2rel_[int(m)].append(int(n))
		self.m2u = movie2rel_

		self.entTotal = np.max([i for i in self.m2u]) + 1
		self.relTotal = np.max([i for i in self.u2m]) + len(self.rel2ids) + 1

	def get_ent_total(self):
		return self.entTotal

	def set_sample_node_neighbour(self, sample_node_neighbour):
		self.sample_node_neighbour = sample_node_neighbour

	def set_rel_type(self, rel_type):
		self.rel_type = rel_type

	def get_rel_type(self):
		return self.rel_type

	def get_rel_total(self):
		return self.relTotal

	def set_lmbda(self, lmbda):
		self.lmbda = lmbda

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def set_opt_method(self, method):
		self.opt_method = method

	def set_test_link_prediction(self, flag):
		self.test_link_prediction = flag

	def set_test_triple_classification(self, flag):
		self.test_triple_classification = flag

	def set_log_on(self, flag):
		self.log_on = flag

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_in_path(self, path):
		self.in_path = path

	def set_out_files(self, path):
		self.out_path = path

	def set_bern(self, bern):
		self.bern = bern

	def set_dimension(self, dim):
		self.hidden_size = dim
		self.ent_size = dim
		self.rel_size = dim

	def set_ent_dimension(self, dim):
		self.ent_size = dim

	def set_rel_dimension(self, dim):
		self.rel_size = dim

	def set_train_times(self, times):
		self.train_times = times

	def set_sample_node_number(self, sample_node_number):
		self.sample_node_number = sample_node_number

	def set_margin(self, margin):
		self.margin = margin

	def set_work_threads(self, threads):
		self.workThreads = threads

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_import_files(self, path):
		self.importName = path

	def set_export_files(self, path, steps = 0):
		self.exportName = path
		self.export_steps = steps

	def set_export_steps(self, steps):
		self.export_steps = steps

	def set_early_stopping(self, early_stopping):
		self.early_stopping = early_stopping

	def sample_data(self, M_IDs):
		total_triples = []
		for M_ID in M_IDs:
			total_movies = set(self.movie2rels.MovieID)
			movie = self.movie2rels.iloc[M_ID]
			posi_tails = []
			nega_tails = []
			total_used = 0
			movie.Genres = eval(movie.Genres)
			gen_num = len(movie.Genres)
			for index,label in enumerate(movie.Genres):
				if index == gen_num - 1:
					lab_number = self.sample_node_neighbour - total_used
				else:
					#percent = movie.Genres[label]
					percent = 1/len(movie.Genres)
					lab_number = int(percent * self.sample_node_neighbour)
				total_used += lab_number
				total_label_movies = self.rel2movie[label]
				positive_tails = random.choices(total_label_movies, k = lab_number)
				total_other_movies = total_movies.difference(set(total_label_movies))
				negative_tails = random.sample(total_other_movies, lab_number * self.negative_ent)
				for posi in positive_tails:
					posi_tails.append([M_ID, posi, label])
				for nega in negative_tails:
					nega_tails.append([M_ID, nega, label])
			total_triples.append([[posi_tails, nega_tails]])
		return total_triples

	def process_smapling_(self, max_rel, max_neigh, sampled_num, negative_ent):
	    movies = random.choices(list(self.m2u.keys()), k = sampled_num)
	    total_triples = []
	    for movie in movies:
	        total_movies = set(list(self.m2u.keys()))
	        pos_triples = []
	        neg_triples = []
	        rels = self.m2u[movie]
	        #print(rels)
	        if len(rels) > max_rel:
	            selected_rels = random.choices(rels, k = max_rel)
	        else:
	            selected_rels = rels
	        len_rel = len(selected_rels)
	        total_ = 0
	        percent = 1/len_rel
	        for index,rel in enumerate(selected_rels):
	            sampled = self.u2m[rel]
	            if index != len_rel - 1:
	                sampled_nei = int(percent * max_neigh)
	                total_ += sampled_nei
	                pos_tails = random.choices(sampled, k = sampled_nei)
	            else:
	                sampled_nei = max_neigh - total_
	                pos_tails = random.choices(sampled, k = sampled_nei)
	            total_other_movies = total_movies.difference(set(sampled))
	            negative_tails = random.sample(total_other_movies, sampled_nei * negative_ent)
	            #print("len", len(negative_tails))
	            for pos_tail in pos_tails:
	                pos_triples.append([movie, pos_tail, rel])
	            for neg_tail in negative_tails:
	                neg_triples.append([movie, neg_tail, rel])
	        total_triples.append([pos_triples, neg_triples])
	    return total_triples
	# call C function for sampling
	def sampling(self):
		#print(self.sample_node_number)
		sampled_nodes = random.choices(range(self.entTotal), k = self.sample_node_number)
		pos_neg_list = self.sample_data(sampled_nodes)

		for r in pos_neg_list:
			for i in r:
				for j in i:
					for q in j:
						#print("self.rel2ids", self.rel2ids)
						#print("j[2]", q[2])
						q[2] = self.rel2ids[q[2]]

		pos_neg_list_2 = self.process_smapling_(self.max_rel, self.sample_node_neighbour, self.sample_node_number, self.negative_ent)

		head_pos = []
		head_neg = []
		tail_pos = []
		tail_neg = []
		relation_pos = []
		rel_type_pos = []
		relation_neg = []
		rel_type_neg = []
		y_pos = []
		y_neg = []

		for elem in pos_neg_list_2:
			pos_triples = elem[0]
			neg_triples = elem[1]
			for pos in pos_triples:
				head_pos.append(pos[0])
				tail_pos.append(pos[1])
				relation_pos.append(pos[2])
				rel_type_pos.append(0)
				y_pos.append(1)
			#print(len(neg_triples))
			for neg in neg_triples:
				#print("len(neg)",len(neg))
				head_neg.append(neg[0])
				tail_neg.append(neg[1])
				relation_neg.append(neg[2])
				rel_type_neg.append(0)
				y_neg.append(-1)

		for elem in pos_neg_list:
			pos_triples = elem[0][0]
			neg_triples = elem[0][1]
			for pos in pos_triples:
				head_pos.append(pos[0])
				tail_pos.append(pos[1])
				relation_pos.append(pos[2])
				rel_type_pos.append(1)
				y_pos.append(1)
			for neg in neg_triples:
				head_neg.append(neg[0])
				tail_neg.append(neg[1])
				relation_neg.append(neg[2])
				rel_type_neg.append(1)
				y_neg.append(-1)

		head_pos = head_pos + head_neg
		tail_pos = tail_pos + tail_neg
		relation_pos = relation_pos + relation_neg
		rel_type_pos = rel_type_pos + rel_type_neg
		y_pos = y_pos + y_neg

		self.batch_h = np.array(head_pos, dtype = np.int64)
		self.batch_r = np.array(relation_pos, dtype = np.int64)
		self.batch_type_r = np.array(rel_type_pos, dtype = np.int64)
		self.batch_t = np.array(tail_pos, dtype = np.int64)
		self.batch_y = np.array(y_pos, dtype = np.float32)
		#self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)

	# save model
	def save_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.save(self.sess, self.exportName)

	# restore model
	def restore_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.restore(self.sess, self.importName)

	def export_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.save(self.sess, self.exportName)
				else:
					self.saver.save(self.sess, path)

	def import_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.restore(self.sess, self.importName)
				else:
					self.saver.restore(self.sess, path)

	def get_parameter_lists(self):
		return self.trainModel.parameter_lists

	def get_parameters_by_name(self, var_name):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					return self.sess.run(self.trainModel.parameter_lists[var_name])
				else:
					return None

	def get_parameters(self, mode = "numpy"):
		res = {}
		lists = self.get_parameter_lists()
		for var_name in lists:
			if mode == "numpy":
				res[var_name] = self.get_parameters_by_name(var_name)
			else:
				res[var_name] = self.get_parameters_by_name(var_name).tolist()
		return res

	def save_parameters(self, path = None):
		if path == None:
			path = self.out_path
		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	def set_parameters_by_name(self, var_name, tensor):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					self.trainModel.parameter_lists[var_name].assign(tensor).eval()

	def set_parameters(self, lists):
		for i in lists:
			self.set_parameters_by_name(i, lists[i])

	def set_model(self, model):
		self.model = model
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.sess = tf.Session()
			with self.sess.as_default():
				initializer = tf.contrib.layers.xavier_initializer(uniform = True)
				with tf.variable_scope("model", reuse=None, initializer = initializer):
					self.trainModel = self.model(config = self)
					if self.optimizer != None:
						pass
					elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
						self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.alpha, initial_accumulator_value=1e-20)
					elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
						self.optimizer = tf.train.AdadeltaOptimizer(self.alpha)
					elif self.opt_method == "Adam" or self.opt_method == "adam":
						self.optimizer = tf.train.AdamOptimizer(self.alpha)
					else:
						self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
					grads_and_vars = self.optimizer.compute_gradients(self.trainModel.loss)
					self.train_op = self.optimizer.apply_gradients(grads_and_vars)
				self.saver = tf.train.Saver()
				self.sess.run(tf.global_variables_initializer())

	def train_step(self, batch_h, batch_t, batch_r, batch_type_r, batch_y):
		feed_dict = {
			self.trainModel.batch_h: batch_h,
			self.trainModel.batch_t: batch_t,
			self.trainModel.batch_r: batch_r,
			self.trainModel.batch_y: batch_y,
			self.trainModel.batch_type_r: batch_type_r
		}
		_, loss = self.sess.run([self.train_op, self.trainModel.loss], feed_dict)
		"""
		print("_p_score", _p_score)
		print("p_score", p_score)
		print("_n_score", _n_score)
		print("n_score", n_score)
		print("n_t", n_t)
		print("p_t", p_t)
		print("p_t - n_t", p_t - n_t)
		print("p_score - n_score", p_score - n_score)
		"""
		#print("p_type_r", p_type_r)
		return loss

	def test_step(self, test_h, test_t, test_r):
		feed_dict = {
			self.trainModel.predict_h: test_h,
			self.trainModel.predict_t: test_t,
			self.trainModel.predict_r: test_r,
		}
		predict = self.sess.run(self.trainModel.predict, feed_dict)
		return predict

	def run(self):
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				if self.early_stopping is not None:
					patience, min_delta = self.early_stopping
					best_loss = np.finfo('float32').max
					wait_steps = 0
				for times in range(self.train_times):
					loss = 0.0
					t_init = time.time()
					for batch in range(self.nbatches):
						self.sampling()
						loss_ = self.train_step(self.batch_h, self.batch_t, self.batch_r, self.batch_type_r, self.batch_y)
						loss += loss_
					t_end = time.time()
					if self.log_on:
						print('Epoch: {}, loss: {}, time: {}'.format(times, loss, (t_end - t_init)))
					if self.exportName != None and (self.export_steps!=0 and times % self.export_steps == 0):
						self.save_tensorflow()
					if self.early_stopping is not None:
						if loss + min_delta < best_loss:
							best_loss = loss
							wait_steps = 0
						elif wait_steps < patience:
							wait_steps += 1
						else:
							print('Early stopping. Losses have not been improved enough in {} times'.format(patience))
							break
				if self.exportName != None:
					self.save_tensorflow()
				if self.out_path != None:
					self.save_parameters(self.out_path)
