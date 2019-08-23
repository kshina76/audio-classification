# audio-classification
##dnn_reco_lowdim
n_mfcc=20  
mfccを時間方向に平均値をとって、1次元ベクトルに  
##dnn_reco_highdim  
loss: 5.6667 - acc: 0.5868  
n_mfcc=40  
mfccの高次まで取得してみた  
高次まで取得すると、声道の音響特性を除去して、ピッチを得ることができる。  
ここで、lossがようやく下がり始め、accも向上し始めた。  
##dnn_reco_various_feature ver1
loss: 11.1997 - acc: 0.3051  
とにかく色々な特徴量を試してみた  
193->256->256->3  
##dnn_reco_various_feature ver2
loss: 10.4656 - acc: 0.3507  
ver3のレイヤーを減らしただけ  
193->256->3  
##dnn_reco_various_feature ver3
loss: 0.5325 - acc: 0.8036  
ユニット数と活性化関数を最適化してみた  
193->280(tanh)->300(relu)  