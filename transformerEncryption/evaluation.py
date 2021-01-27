#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize


# In[2]:


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# In[3]:


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


# In[4]:


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


# In[5]:


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


# In[6]:


def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


# In[7]:


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=gelu),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


# In[8]:


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2


# In[9]:


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)


  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2


# In[10]:


class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.w_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.p_embedding = tf.keras.layers.Embedding(maximum_position, d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, p, training, mask):

    # adding embedding and position encoding.
    x = self.w_embedding(x)  # (batch_size, input_seq_len, d_model)
    p = self.p_embedding(p)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += p[:x.shape[1],:]
    x = self.layernorm(x)
    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)


# In[11]:


class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.w_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.p_embedding = tf.keras.layers.Embedding(maximum_position, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, p, enc_output, training, 
           look_ahead_mask, padding_mask):
    
    attention_weights = {}

    x = self.w_embedding(x)  # (batch_size, input_seq_len, d_model)
    p = self.p_embedding(p)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += p[:x.shape[1],:]
    x = self.layernorm(x)
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


# In[12]:


class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, input_len, target_len, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, input_len, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, target_len, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inp, tar, pos, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
    
    enc_output = self.encoder(inp, pos, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, pos, enc_output, training, look_ahead_mask, dec_padding_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights


# In[13]:


c_vocab = pickle.load( open( "corpus_dict.p", "rb" ) )
e_vocab = pickle.load( open( "cypher_dict.p", "rb" ) )
r_c_vocab = pickle.load( open( "corpus_rev_dict.p", "rb" ) )
r_e_vocab = pickle.load( open( "cypher_rev_dict.p", "rb" ) )


# In[14]:


num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
input_vocab_size = len(c_vocab) + 1
target_vocab_size = len(e_vocab) + 1
max_len=31
dropout_rate = 0.2


# In[15]:


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# In[16]:


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)


# In[17]:


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
  


# In[18]:


encrypter = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          input_len=max_len, 
                          target_len=max_len,
                          rate=dropout_rate)
decrypter = Transformer(num_layers, d_model, num_heads, dff,
                        target_vocab_size,
                        input_vocab_size, 
                        input_len=max_len, 
                        target_len=max_len,
                        rate=dropout_rate)


# In[19]:


encrypter.compile(loss=loss_function, optimizer=optimizer)
decrypter.compile(loss=loss_function, optimizer=optimizer)


# In[20]:


def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask


# In[21]:


en_ckpt_path = "./checkpoints/encrypter"

en_ckpt = tf.train.Checkpoint(transformer=encrypter,
                           optimizer=optimizer)

en_ckpt_manager = tf.train.CheckpointManager(en_ckpt, en_ckpt_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if en_ckpt_manager.latest_checkpoint:
  en_ckpt.restore(en_ckpt_manager.latest_checkpoint)
  print ('Encrypter loaded.')


# In[22]:


de_ckpt_path = "./checkpoints/decrypter3"

de_ckpt = tf.train.Checkpoint(transformer=decrypter,
                           optimizer=optimizer)

de_ckpt_manager = tf.train.CheckpointManager(de_ckpt, de_ckpt_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if de_ckpt_manager.latest_checkpoint:
  de_ckpt.restore(de_ckpt_manager.latest_checkpoint)
  print ('Decrypter loaded.')


# In[23]:


def find(word, corpus):
    if word not in corpus:
        wo = []
        num = []
        wo.append("UNK")
        num.append(corpus["UNK"])
        for char in word:
            wo.append(char)
            num.append(corpus[char])
        wo.append("UNK")
        num.append(corpus["UNK"])
        word = wo
    else:
        num = corpus[word]
    return word, num


# In[24]:


def reformat(line, corpus_r):
    l = []
    li = []
    for word in line:
        w, i = find(word, corpus_r)
        if type(w) == list:
            l.extend(w)
            li.extend(i)
        else:
            l.append(w)
            li.append(i)
    return l, li


# In[25]:


def cleanTokenize(sequence,vocab):
  isString = isinstance(sequence, str)
  if isString:
    sequence = re.sub("[^\x41-\x7A]+", ' ', sequence)
    sequence = re.sub("[\x5B-\x60]+", ' ', sequence)
    sequence = word_tokenize(sequence.lower())
  inp_sen, inp = reformat(sequence, vocab)
  return inp, inp_sen


# In[26]:


def encrypt(sen_token, max_len, pos):

  encoder_input = tf.expand_dims(sen_token, 0)

  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [1]
  output = tf.expand_dims(decoder_input, 0)

  for i in range(max_len-2):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
    
    #import pdb;pdb.set_trace()

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = encrypter(encoder_input, 
                                                 output,
                                                 pos,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    output = tf.concat([output, predicted_id], axis=-1)

    # return the result if the predicted_id is equal to the end token
    if predicted_id == 2:
      break

    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
  return tf.squeeze(output, axis=0), attention_weights


# In[27]:


def decrypt(encryption, max_len, pos):

  encoder_input = tf.expand_dims(encryption, 0)

  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [1]
  output = tf.expand_dims(decoder_input, 0)

  for i in range(max_len-2):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
    
    #import pdb;pdb.set_trace()

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = decrypter(encoder_input, 
                                                 output,
                                                 pos,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    output = tf.concat([output, predicted_id], axis=-1)

    # return the result if the predicted_id is equal to the end token
    if predicted_id == 2:
      break
    
  return tf.squeeze(output, axis=0), attention_weights


# In[28]:


def plot_attention_weights(attention, sentence, result, layer):
  fig = plt.figure(figsize=(16, 8))

  attention = tf.squeeze(attention[layer], axis=0)

  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)

    # plot the attention weights
    ax.matshow(attention[head][:-1, :], cmap='viridis')

    fontdict = {'fontsize': 10}

    ax.set_xticks(range(len(sentence)+2))
    ax.set_yticks(range(len(result)))

    ax.set_ylim(len(result)-1.5, -0.5)

    ax.set_xticklabels(
        ['<start>']+[w for w in sentence]+['<end>'], 
        fontdict=fontdict, rotation=90)

    ax.set_yticklabels([i for i in result], 
                       fontdict=fontdict)

    ax.set_xlabel('Head {}'.format(head+1))

  plt.tight_layout()
  plt.show()


# In[29]:


def evaluation(sample, max_len, vocab, r_vocab, plot='', print_out=True):
  sen_token = sample[0][:7]
  en_result, en_attention_weights = encrypt(sen_token, max_len, sample[2][:7])
  encrypt_res = en_result.numpy()
  encryption = [i for i in encrypt_res if i > 1]

  de_result, de_attention_weights = decrypt(encrypt_res, max_len, sample[2][:7])
  decrypt_res = de_result.numpy()
  decryption = [r_vocab[i] for i in decrypt_res if i > 1]

  if print_out:
    print('Input: ')
    for w in sample[0]:
      if w == 0:
        break
      print(r_c_vocab[w], end=' ')
    print('\nEncryption: {}'.format(encrypt_res))
    print('Decryption: {}'.format(decryption))

  if plot:
    plot_attention_weights(en_attention_weights, [w for w in sample[0] if w!= 0 and w>2], encryption, plot)
    plot_attention_weights(de_attention_weights, encryption[:-1], decryption, plot)


# In[30]:


def manage_input(usr_str,max_len,vocab):
  sen_token, sentence = cleanTokenize(usr_str,vocab)
  
  pad_sen = np.zeros(max_len-1,dtype=np.int32)
  pos = np.zeros(max_len-1,dtype=np.int32)
  pad_sen[0]=1
  pos[0]=1
  for i,t in enumerate(sen_token):
    pad_sen[i+1]=t
    pos[i+1]=pos[i]+1
  pad_sen[len(sen_token)+1]=2
  pos[len(sen_token)+1]=pos[len(sen_token)]+1
  return pad_sen,pos


# In[31]:


def demo(max_len, vocab, r_vocab):
  while True:
    usr_str = input('Enter: ')
    sen_token,pos = manage_input(usr_str,max_len,vocab)
    en_result, en_attention_weights = encrypt(sen_token, max_len, pos)
    encrypt_res = en_result.numpy()
    encryption = [i for i in encrypt_res if i > 0]

    de_result, de_attention_weights = decrypt(en_result, max_len, pos)
    decrypt_res = de_result.numpy()
    decryption = [r_vocab[i] for i in decrypt_res if i > 0]

    print('Encryption: {}'.format(encrypt_res))
    print('Decryption: {}'.format(decryption))


# In[32]:


demo(31, c_vocab, r_c_vocab)


# In[ ]:


with open('test.npy','rb') as f:
  test_set = np.load(f)


# In[ ]:


evaluation(test_set[2],31, c_vocab, r_c_vocab, plot='decoder_layer6_block2')


# In[ ]:


test_set[2]


# In[ ]:


def test_step(sentence, pos, max_len):  
  en_result, en_attention_weights = encrypt(sentence, max_len, pos)

  de_result, de_attention_weights = decrypt(en_result, max_len, pos)

  for i in range(max_len-de_result.shape[0]-1):
    de_result=tf.concat([de_result, [0]], axis=-1)
  
  mask = tf.math.not_equal(pos,0)

  positives = tf.cast(tf.math.equal(sentence,de_result), dtype=tf.float32)
  positives = tf.multiply(positives, tf.cast(mask, dtype=tf.float32))
  accuracy = tf.reduce_sum(positives)/tf.boolean_mask(positives,mask).shape[0]
  return accuracy


# In[ ]:


def test(max_len,vocab,accuracies):
  with open('test.npy','rb') as f:
    test_set = np.load(f)
  for i,sample in enumerate(test_set[:5000]):
    if i%100 == 0:
      print(i,5000)
    accuracies.append(test_step(sample[0],sample[2],max_len))
    with open('test_acc.p','wb') as f:
      pickle.dump(accuracies,f)
  return sum(accuracies)/len(accuracies)


# In[ ]:


accuracies=[]
test(21,c_vocab,accuracies)


# In[ ]:


demo(21,c_vocab,r_c_vocab)


# In[ ]:


with open('test_acc.p','rb') as f:
  accuracies = pickle.load(f)


# In[ ]:


sum(accuracies)/len(accuracies)


# In[ ]:


high=0
low=0
for acc in accuracies[:3000]:
  if acc > 0.9:
    high+=1
  else:
    low+=1
plt.bar([1,2],[high,low],width=0.5,tick_label=['90%+','90%-'])
plt.show


# In[ ]:




