#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    self.wq = tf.keras.layers.Dense(d_model,activation=gelu,kernel_regularizer='l2')
    self.wk = tf.keras.layers.Dense(d_model,activation=gelu,kernel_regularizer='l2')
    self.wv = tf.keras.layers.Dense(d_model,activation=gelu,kernel_regularizer='l2')

    self.dense = tf.keras.layers.Dense(d_model,activation=gelu,kernel_regularizer='l2')

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
      tf.keras.layers.Dense(dff, activation=gelu, kernel_initializer='he_uniform', activity_regularizer='l2'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model, activation=gelu, kernel_initializer='he_uniform', activity_regularizer='l2')  # (batch_size, seq_len, d_model)
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

    self.w_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, embeddings_initializer='he_uniform')
    self.p_embedding = tf.keras.layers.Embedding(maximum_position, d_model, embeddings_initializer='he_uniform')

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, p, training, mask):

    # adding embedding and position encoding.
    x = self.w_embedding(x)  # (batch_size, input_seq_len, d_model)
    p = self.p_embedding(p)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += p
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

    self.w_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, embeddings_initializer='he_uniform')
    self.p_embedding = tf.keras.layers.Embedding(maximum_position, d_model, embeddings_initializer='he_uniform')

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
    x += p[:,:-1,:]
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


# In[15]:


with open('train.npy', 'rb') as f:
    dataset = np.load(f)

c_vocab = pickle.load( open( "corpus_dict.p", "rb" ) )
e_vocab = pickle.load( open( "cypher_dict.p", "rb" ) )


# In[16]:


def train_val(dataset):
  np.random.shuffle(dataset)
  train_set=dataset[:int(dataset.shape[0]*0.99)]
  val_set=dataset[int(dataset.shape[0]*0.99):]

  val_data=[]
  for v_d in val_set:
    val_data.append(([v_d[0]],[v_d[1]],[v_d[2]]))

  return train_set, np.array(val_data)


# In[17]:


def data_split(dataset,batch_size):
  train_data=[]
  np.random.shuffle(dataset)
  for i in range(dataset.shape[0]//batch_size+1):
    try:
      train_data.append((dataset[i*batch_size:(i+1)*batch_size,0],
                         dataset[i*batch_size:(i+1)*batch_size,1],
                         dataset[i*batch_size:(i+1)*batch_size,2]))
    except(IndexError):
      train_data.append((dataset[i*batch_size:,0],
                         dataset[i*batch_size:,1],
                         dataset[i*batch_size:,2]))
  return train_data


# In[18]:


num_layers = 6
d_model = 512
dff = 2048
num_heads = 8

input_vocab_size = len(e_vocab) + 1
target_vocab_size = len(c_vocab) + 1
max_len=31
dropout_rate = 0.4


# In[19]:


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


# In[20]:


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


# In[21]:


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


# In[22]:


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


# In[23]:


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
validate_loss = tf.keras.metrics.Mean(name='validate_loss')
validate_accuracy = tf.keras.metrics.Mean(name='validate_accuracy')


# In[24]:


transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          input_len=max_len,
                          target_len=max_len,
                          rate=dropout_rate)


# In[25]:


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


# In[26]:


checkpoint_path = "./checkpoints/decrypter3"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint)
#   print ('Latest checkpoint restored!!')


# In[27]:


EPOCHS = 20


# In[28]:


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar, pos):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, pos,
                                 True,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))

@tf.function(input_signature=train_step_signature)
def validate_step(inp, tar, pos):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = transformer(inp, tar_inp, pos, False, enc_padding_mask, combined_mask, dec_padding_mask)
    loss = loss_function(tar_real, predictions)

    validate_loss(loss)
    validate_accuracy(accuracy_function(tar_real, predictions))


# In[29]:


train_loss_result=[]
train_accuracy_result=[]
validate_loss_result=[]
validate_accuracy_result=[]
epoch_result=np.zeros((4,20))
train_set,val_data=train_val(dataset)
for epoch in range(EPOCHS):
  start = time.time()
  batch_start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()
  validate_loss.reset_states()
  validate_accuracy.reset_states()

  train_data=data_split(train_set,32)

  for (batch, (tar, inp, pos)) in enumerate(train_data):
    train_step(inp, tar, pos)

    if batch % 50 == 0:
      print ('Epoch {} Batch {}/{} Loss {:.4f} Accuracy {:.4f} b_time {:.4f}'.format(
          epoch + 1, batch+1, len(train_data), train_loss.result(), train_accuracy.result(), time.time()-batch_start))
      train_loss_result.append(train_loss.result())
      train_accuracy_result.append(train_accuracy.result())
      batch_start = time.time()

  for (batch, (tar, inp, pos)) in enumerate(val_data):
    validate_step(inp, tar, pos)
    validate_loss_result.append(validate_loss.result())
    validate_accuracy_result.append(validate_accuracy.result())

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                        ckpt_save_path))

  epoch_result[0,epoch]=tf.reduce_mean(train_loss_result)
  epoch_result[1,epoch]=tf.reduce_mean(train_accuracy_result)
  epoch_result[2,epoch]=tf.reduce_mean(validate_loss_result)
  epoch_result[3,epoch]=tf.reduce_mean(validate_accuracy_result)

  print ('Epoch {} Loss {:.4f} Accuracy {:.4f} v_loss {:.4f} v_acc {:.4f}'.format(epoch + 1,
                                                                                  epoch_result[0,epoch],
                                                                                  epoch_result[1,epoch],
                                                                                  epoch_result[2,epoch],
                                                                                  epoch_result[3,epoch]))

  print ('Time taken for 1 epoch: {:.2f} secs\n'.format(time.time() - start))


# In[30]:


plt.plot(train_accuracy_result)

plt.show()


# In[31]:


plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training_Validation_Accuracy")
plt.plot()
x=np.arange(1,21)
plt.plot(x,epoch_result[1,:],label='train_acc')
plt.plot(x,epoch_result[3,:],label='validation_acc')

plt.legend()
plt.show()


# In[32]:


plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training_Validation_Loss")
plt.plot()
x=np.arange(1,20)
plt.plot(x,epoch_result[0,:],label='train_loss')
plt.plot(x,epoch_result[2,:],label='validation_loss')

plt.legend()
plt.show()


# In[ ]:


# with open('logs/decrypter_loss_accuracy.pulickle','wb') as de_file:
    # pickle.dump([train_loss_result,train_accuracy_result,validate_loss_result,validate_accuracy_result],de_file)


# In[ ]:


def evaluate(inp_sentence):
  start_token = [tokenizer_pt.vocab_size]
  end_token = [tokenizer_pt.vocab_size + 1]

  # inp sentence is portuguese, hence adding the start and end token
  inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)

  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [tokenizer_en.vocab_size]
  output = tf.expand_dims(decoder_input, 0)

  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input,
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer_en.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights

    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights


# In[ ]:


def plot_attention_weights(attention, sentence, result, layer):
  fig = plt.figure(figsize=(16, 8))

  sentence = tokenizer_pt.encode(sentence)

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
        ['<start>']+[tokenizer_pt.decode([i]) for i in sentence]+['<end>'],
        fontdict=fontdict, rotation=90)

    ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
                        if i < tokenizer_en.vocab_size],
                       fontdict=fontdict)

    ax.set_xlabel('Head {}'.format(head+1))

  plt.tight_layout()
  plt.show()


# In[ ]:
