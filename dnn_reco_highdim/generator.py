import numpy as np
import os
import glob
import librosa
import random
import math
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model, Model
from keras.utils import Sequence
from keras.utils import np_utils

class BatchGenerator(Sequence):

	def __init__(self, data_path, labels, batch_size, sample_rate, sample_length, threshold):
		self.batch_size = batch_size
		self.data_path = data_path
		self.labels = labels
		self.sample_rate = sample_rate
		self.sample_length = sample_length
		self.threshold = threshold
		self.length = len(data_path)
		self.batches_per_epoch = math.ceil(self.length / batch_size)

	def preprocess(self, audio):
		audio, _ = librosa.effects.trim(audio, self.threshold)

		# すべての音声フェイルを指定した同じサイズに変換
		if self.threshold is not None:
			if len(audio) <= self.sample_length:
				# padding
				pad = self.sample_length - len(audio)
				audio = np.concatenate((audio, np.zeros(pad, dtype=np.float32)))
			else:
				# trimming
				start = random.randint(0, len(audio) - self.sample_length - 1)
				audio = audio[start:start + self.sample_length]
			mfccs = librosa.feature.mfcc(audio, sr=self.sample_rate, n_mfcc=40)
			# mfccを時間軸方向で平均値をとって、時系列データじゃなくする。（もしそのまま学習したいならCNNやLSTMで時系列を扱えるモデルで。）
			mfccs = np.mean(mfccs, axis=1)

		return mfccs

	def __getitem__(self, idx):
		batch_from = self.batch_size * idx
		batch_to = batch_from + self.batch_size

		if batch_to > self.length:
			batch_to = self.length

		x_batch = []  # feature
		y_batch = []  # target

		for i in range(batch_from, batch_to):
			audio, _ = librosa.load(self.data_path[i], sr=self.sample_rate)
			mfccs = self.preprocess(audio)

			x_batch.append(mfccs)
			y_batch.append(self.labels[i])

		x_batch = np.asarray(x_batch)
		y_batch = np.asarray(y_batch)

		return x_batch, y_batch

	def __len__(self):
		return self.batches_per_epoch


	def on_epoch_end(self):
		pass

