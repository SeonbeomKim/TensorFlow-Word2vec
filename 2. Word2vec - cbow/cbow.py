#https://www.tensorflow.org/tutorials/word2vec
import tensorflow as tf # 1.4
import collections
import numpy as np
import os

window_size = 4
number_of_word = 50000
embedding_size = 128
negative_sample = 64 #softmax 할 때, number_of_word 전부 다 사용하면 너무 계산량이 많음, 따라서 거리가 먼 샘플 64개 지정.
train_rate = 0.0001


def draw_most_word_pyplot(outname, most):
	from sklearn.manifold import TSNE #pip install scipy, scikit-learn
	import matplotlib.pyplot as plt #pip install matplotlib

	most_common_word = rev_word_table[:most]
	most_common_embedding = embedding[:most].eval(session=sess)

	plt.figure(figsize=(18,18))
	tsne = TSNE(perplexity = 30, n_components = 2, init='pca')
	low_dim_embed = tsne.fit_transform(most_common_embedding)

	for i, label in enumerate(most_common_word):
		x, y = low_dim_embed[i]
		plt.scatter(x,y)
		plt.annotate(label, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')

	plt.savefig(outname)
	plt.close()


#두 단어간의 코사인 유사도
def cosine_similarity_with_two_word(word1, word2):
	word1_embedding = embedding[word_table[word1]].eval(session=sess)
	word2_embedding = embedding[word_table[word2]].eval(session=sess)
	return np.sum(word1_embedding*word2_embedding)/ ( np.sqrt(np.sum(word1_embedding*word1_embedding)) * np.sqrt(np.sum(word2_embedding*word2_embedding)) )


#상위 most_num 개수랑 비교해서 내가 입력한 단어와 유사도가 높은것 want_to_see_num 개 출력.
def cosine_similarity_with_most_word(word, most=number_of_word, want_to_see_num=number_of_word):
	inputindex = word_table[word]
	word_embedding = embedding[inputindex].eval(session=sess)
	word2_embedding = embedding[:most].eval(session=sess)

	result = (np.sum(word_embedding * word2_embedding, axis=1) / (  np.sqrt(np.sum(word_embedding*word_embedding)) * np.sqrt(np.sum(word2_embedding*word2_embedding, axis=1)) ) ).tolist()
	result = list(zip(rev_word_table[:most], result))
	result.sort(key=lambda x:x[1], reverse=True)

	return np.array(result)[:want_to_see_num]


# input은 batch * number_of_word shape로, target은 number_of_word * 1 shape로 
def make_batch_dataset(word, start, end): 
	input_ = []
	target_ = []
	
	if end == -1 or end > len(word):
		end = len(word)
	
	for i in range(start, end):
		left = i-window_size
		if left < 0:
			left = 0

		#i 제외하고 양옆으로 window_size개만 뽑음
		leftset = word[left:i]
		rightset = word[i+1:i+window_size+1] 
		
		#좌우 개수가 동일할때.
		if len(leftset) == len(rightset):
			input_.append(leftset+rightset)
			target_.append([word[i]])			

	return input_, target_


# 단어 빈도 상위권 뽑아서 단어별로 0부터 X-2 까지 넘버 부여하고, X-1는 UNK로 부여.
def preprocess(path): 
	with open(path, 'r') as f:
		word = (f.readline().split())	#text8은 하나의 줄이며 단어마다 띄어쓰기로 구분.
	
	table = collections.Counter(word).most_common(number_of_word-1) #빈도수 상위 x-1개 뽑음. 튜플형태로 정렬되어있음 [("단어", 빈도수),("단어",빈도수)] 
	word_table = {i[0]:index for index, i in enumerate(table)} # i is key. 상위 x-1개에 대해서 0부터 x-2까지 매핑시킴. 
	word_table['UNK'] = number_of_word-1 # 이로써 word_table 크기는 number_of_word 변수 크기.

	#cosine_similarity_with_most_word, draw_most_word_pyplot 함수에서 사용
	rev_word_table = [i[0] for i in table] #단어만 상위권부터 넣음 => [단어, 단어, 단어, 단어 ... 'unk']
	rev_word_table.append('UNK')

	for i in range(len(word)): # word는 word_table에 의해서 숫자로 맵핑됨
		if word[i] in word_table:
			word[i] = word_table[word[i]]
		else:
			word[i] = word_table['UNK']

	return word, word_table, rev_word_table


def train(word):
	loss = 0
	batch_word_size = 128 # 한번에 학습시킬 단어 개수

	for i in range( int(np.ceil(len(word)/batch_word_size)) ):
		#print(i+1, ' / ', int(np.ceil(len(word)/batch_word_size)))
		input_, target_ = make_batch_dataset(word, batch_word_size * i, batch_word_size * (i + 1))
		train_loss, _ = sess.run([nce_loss, minimize], {X:input_, Y:target_})
		loss += train_loss
	
	return loss


def run(word, restore = 0):
	if restore != 0:
		saver.restore(sess, "./saver/"+str(restore)+".ckpt") #weight 복구.
		#draw_most_word_pyplot('./draw/'+str(restore), 500) # 상위 500단어에 대해서 벡터 시각화 파일 생성해라.
	
	for epoch in range(restore+1, 100001):
		train_loss = train(word)
		print("epoch : ", epoch, " train_loss : ", train_loss)

		summary = sess.run(merged, {train_loss_tensorboard:train_loss})
		writer.add_summary(summary, epoch)

		if epoch % 1 == 0:
			if not os.path.isdir('./saver'):
				os.mkdir('./saver')
			save_path = saver.save(sess, './saver/'+str(epoch)+".ckpt") #weight 저장.

			if not os.path.isdir('./draw'):
				os.mkdir('./draw')			
			draw_most_word_pyplot('./draw/'+str(epoch), 500) # 상위 500단어에 대해서 벡터 시각화 파일 생성해라.


#with tf.device('/gpu:0'):
X = tf.placeholder(tf.int32, (None, None)) #batch * window_size*2
Y = tf.placeholder(tf.int32, (None, 1)) # batch * number_of_word * 1

embedding = tf.Variable(tf.random_uniform([number_of_word, embedding_size], -1., 1.))
output_of_hidden = tf.nn.embedding_lookup(embedding, X) # batchsize * windowsize*2 * embedding_size
mean_output = tf.reduce_mean(output_of_hidden, axis=1) # windowsize*2 애들끼리 평균내라. ==> batchsize * embedding_size

#https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/nn/nce_loss
nce_weight = tf.get_variable("nce_weight", shape=[number_of_word, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
nce_bias = tf.Variable(tf.zeros([number_of_word]))

nce_loss = tf.nn.nce_loss(weights = nce_weight, biases = nce_bias, labels = Y, inputs = mean_output, num_sampled = negative_sample, num_classes = number_of_word)
nce_loss = tf.reduce_mean(nce_loss)

optimizer = tf.train.AdamOptimizer(train_rate)
minimize = optimizer.minimize(nce_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#텐서보드 실행 tensorboard --logdir=./tensorboard/. # 띄어쓰기 조심. logdir부터 쭉 다 붙여써야함.
train_loss_tensorboard = tf.placeholder(tf.float32)
train_summary = tf.summary.scalar("train_loss", train_loss_tensorboard) 
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./tensorboard', sess.graph)

saver = tf.train.Saver(max_to_keep=10000)
word, word_table, rev_word_table = preprocess('text8') #word = 원본 문장 숫자로 바꾼것, word_table 빈도수높은 단어들 숫자부여한것.

run(word)


'''
##test code

while True:
	saver.restore(sess, "./saver/"+str(103)+".ckpt") #weight 복구.
	print("입력해라")
	print(cosine_similarity_with_two_word(input(), input()))
	print(cosine_similarity_with_most_word(input(), want_to_see_num=20)) #입력단어, 상위1000개랑 비교해서, 유사도출력20개
'''

'''
##make_batch_dataset function test
print(word[:15])
a, b = make_batch_dataset(word, 0, 10)
c = list(zip(a, b))
for i in c:
	print(i)

print(a)
c = sess.run(output_of_hidden, {X:a})
print(np.array(c).shape)
d = sess.run(mean_output, {X:a})
print(np.array(d).shape)
'''