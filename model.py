import tensorflow as tf
import numpy as np
from data_preprocess import read_data
import sys

tf.set_random_seed(7)

class MemN2N(object):
    def __init__(self, config, sess):
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.mem_size = config['mem_size']
        self.is_test = config['test']
        self.n_epoch = config['n_epoch']
        self.n_hop = config['n_hop']
        self.n_words = config['n_words']
        self.lr = config['lr']
        self.std_dev = config['std_dev']
        
        self.inp_X = tf.placeholder('int32',[self.batch_size, self.mem_size])
        self.inp_Y = tf.placeholder('int32', [self.batch_size,])
        self.time = tf.placeholder('int32', [self.batch_size, self.mem_size])    
        self.loss = None  
        self.session = sess

    def init_model(self):
        # Input and output embeddings
        i_emb   = tf.Variable(tf.random_normal([self.n_words, self.emb_dim], stddev=self.std_dev), dtype='float32')
        o_emb   = tf.Variable(tf.random_normal([self.n_words, self.emb_dim], stddev=self.std_dev), dtype='float32')
        # Input and output embedding for time information
        i_emb_T = tf.Variable(tf.random_normal([self.n_words, self.emb_dim], stddev=self.std_dev), dtype='float32')
        o_emb_T = tf.Variable(tf.random_normal([self.n_words, self.emb_dim], stddev=self.std_dev), dtype='float32')
        # Query fixed to a vector of 0.1
        initial_q   = tf.constant(0.1, shape=[self.batch_size, self.emb_dim], dtype='float32')
        # For linear mapping of u between hops
        Aw = tf.Variable(tf.random_normal([self.emb_dim, self.emb_dim], stddev=self.std_dev), dtype='float32')
        Ab = tf.Variable(tf.random_normal([self.emb_dim], stddev=self.std_dev), dtype='float32')
        #For storing the final layer of each hop
        hid = []
        hid.append(initial_q)
        # Final weight matrix
        W = tf.Variable(tf.random_normal([self.n_words, self.emb_dim], stddev=self.std_dev), dtype='float32')

        #Memory vectors
        mem_C = tf.nn.embedding_lookup(i_emb, self.inp_X)
        mem_T = tf.nn.embedding_lookup(i_emb_T, self.time)
        mem = tf.add(mem_C, mem_T)

        #Output vectors
        out_C = tf.nn.embedding_lookup(o_emb, self.inp_X)
        out_T = tf.nn.embedding_lookup(o_emb, self.time)
        out = tf.add(out_C, out_T)

        for hop in range(self.n_hop):
            hid3d = tf.reshape(hid[-1], [-1, self.emb_dim, 1])
            probs = tf.nn.softmax(tf.batch_matmul(mem, hid3d))
            o = tf.batch_matmul(out, probs, adj_x=True)
            sigma_uo = tf.add(hid3d, o)
            hid2d = tf.reshape(sigma_uo, [-1, self.emb_dim])
            Cout = tf.add(tf.matmul(hid2d, Aw), Ab) 
            hid.append(Cout)
        
        z = tf.matmul(hid[-1], W, transpose_b=True)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(z, self.inp_Y)
        

    def train(self, data):
        self.init_model()
        optim = tf.train.AdamOptimizer(self.lr)
        train_op = optim.minimize(self.loss)
        cost = 0

        N = int((len(data)/self.batch_size)+1)
        t = np.ndarray([self.batch_size, self.mem_size])
        
        for x in range(0, self.mem_size):
            t[:, x].fill(x)

        tf.initialize_all_variables().run()
        
        for n in range(N):
            inputs = []
            targets = []
            for item in range(self.batch_size):
                mark = np.random.randint(self.mem_size+1, len(data))
                next_word = data[mark]
                prec_words = data[mark-self.mem_size : mark]
                inputs.append(prec_words)
                targets.append(next_word)
            inputs = np.asarray(inputs)
            targets = np.asarray(targets)
           
            fd = {
                self.inp_X : inputs,
                self.inp_Y : targets,
                self.time : t
            }
           

            _, loss = self.session.run([train_op, self.loss], feed_dict=fd)
            cost += np.sum(loss)
            # sys.stdout.write('.')
            # sys.stdout.flush()
            print "cost",cost/(n*self.batch_size), "Perp:",np.exp(cost/(n*self.batch_size)),"--",n,"/",N 

        print
        print "Perplexity :",np.exp(cost/(N*self.batch_size))
        print 

    def test():
        pass

    # def generate_data(self, data):
    #     for item in range(self.batch_size):
    #         mark = np.random.randint(self.mem_size+1, len(data))
    #         next_word = data[mark]
    #         prec_words = data[mark-self.mem_size : mark]
    #         inputs.append(prec_words)
    #         targets.append(next_word)
