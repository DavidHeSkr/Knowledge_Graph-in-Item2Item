#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

if tf.__version__ > '0.12.1':
	matmul_func = tf.matmul
else:
	matmul_func = tf.batch_matmul

class TransE(Model):
	r'''
	TransE is the first model to introduce translation-based embedding,
	which interprets relations as the translations operating on entities.
	'''
	def _transfer(self, transfer_matrix, embeddings):
		return matmul_func(embeddings, transfer_matrix)

	def _calc(self, h, t, r):
		h = tf.nn.l2_normalize(h, -1)
		t = tf.nn.l2_normalize(t, -1)
		r = tf.nn.l2_normalize(r, -1)
		return abs(h + r - t)

	def embedding_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#Defining required parameters of the model, including embeddings of entities and relations
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.type_transfer_matrix = tf.get_variable(name = "type_transfer_matrix", shape = [config.rel_type, config.ent_size * config.rel_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
								"rel_embeddings":self.rel_embeddings,
                                "type_transfer_matrix":self.type_transfer_matrix}

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
		#The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
		pos_h, pos_t, pos_r, pos_type_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r, neg_type_r = self.get_negative_instance(in_batch = True)
		#Embedding entities and relations of triples, e.g. p_h, p_t and p_r are embeddings for positive triples
		self.p_h = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
		self.p_t = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
		self.p_r = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
		self.p_type_r = tf.reshape(tf.nn.embedding_lookup(self.type_transfer_matrix, pos_type_r), [-1, config.ent_size, config.rel_size])
		self.n_h = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
		self.n_t = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
		self.n_r = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
		self.n_type_r = tf.reshape(tf.nn.embedding_lookup(self.type_transfer_matrix, pos_type_r), [-1, config.ent_size, config.rel_size])
		self.p_h_ = self._transfer(self.p_type_r, self.p_h)
		self.p_t_ = self._transfer(self.p_type_r, self.p_t)
		self.n_h_ = self._transfer(self.n_type_r, self.n_h)
		self.n_t_ = self._transfer(self.n_type_r, self.n_t)
		#Calculating score functions for all positive triples and negative triples
		#The shape of _p_score is (batch_size, 1, hidden_size)
		#The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
		self._p_score = self._calc(self.p_h_, self.p_t_, self.p_r)
		self._n_score = self._calc(self.n_h_, self.n_t_, self.n_r)
		#The shape of p_score is (batch_size, 1, 1)
		#The shape of n_score is (batch_size, negative_ent + negative_rel, 1)
		self.p_score =  tf.reduce_sum(self._p_score, -1, keep_dims = True)
		self.n_score =  tf.reduce_sum(self._n_score, -1, keep_dims = True)
		#Calculating loss to get what the framework will optimize
		self.loss = tf.reduce_mean(tf.maximum(self.p_score - self.n_score + config.margin, 0))

	def predict_def(self):
		predict_h, predict_t, predict_r = self.get_predict_instance()
		predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_h)
		predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_t)
		predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, predict_r)
		self.predict = tf.reduce_mean(self._calc(predict_h_e, predict_t_e, predict_r_e), 1, keep_dims = False)
