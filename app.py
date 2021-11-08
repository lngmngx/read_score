import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def EMA(window,window_size,coef=2):
    alpha = coef/(window_size+1)
    smoothed = [scores[0]]
    for i in range(1,len(window)):
        smoothed.append(window[i]*alpha+smoothed[-1]*(1-alpha))
    #plt.plot(range(1,len(scores)+1),scores)
    #plt.plot(range(len(scores)-len(smoothed)+1,len(scores)+1),smoothed)
    return smoothed[-1]

def smooth(scores,window_size,coef=2):
    Y = []
    for i in range(1,len(scores)+1):
        window = scores[max(0,i-window_size):i]
        Y.append(EMA(window,window_size,coef=coef))
    return Y


np.random.seed(8)
scores = []
for x in np.linspace(5,30,30):
    if np.random.rand()<0.8:
        scores.append(int(max(0,3*x+np.random.randint(-4,5))))
    else:
        if np.random.rand()<0.8:
            scores.append(str(int(3*x*0.1*np.random.randint(1,3))))
        else:
            scores.append(str(int(min(3*x*3,90))))


col1, col2 = st.columns(2)

scores = col1.text_area("历史分数",value=",".join(np.array(scores).astype(str)))
scores = np.array(scores.split(",")).astype(int)


window_size = col2.slider("计算范围",1,20,value=10)

X = range(1,len(scores)+1)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(X,scores,c='black',label='Raw Scores')
ax.plot(X,[np.mean(scores[max(0,i-window_size):i]) for i in range(1,len(scores)+1)],label='Mean')
ax.plot(X,[np.median(scores[max(0,i-window_size):i]) for i in range(1,len(scores)+1)],c='red',label='Median')
ax.plot(X,smooth(scores,window_size=window_size,coef=2),c='green',label='Exponential Moving Average')
ax.set_ylim(0,100)
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Score')
ax.set_title('Smoothed Scores')
st.pyplot(fig)

st.write('_____')
left, middle, right = st.columns(3)
score_0 = left.number_input('低于此分数，开口阅读比例为0',value=45)
score_100 = middle.number_input('高于此分数，开口阅读比例为100',value=70)
n_cap = right.slider('前x次开口阅读比例为0',1,20,value=10)

def read_time(x):
    a = 100/(score_100-score_0)
    b = -score_0*a
    return min(max(0,a*x+b),100)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(X,scores,c='black',label='Raw Scores')
ax.plot(X,np.append(np.zeros(n_cap),[read_time(np.mean(scores[max(0,i-window_size):i])) for i in range(1,len(scores)+1)][n_cap:]),label='Speaking Ratio (Mean)')
ax.plot(X,np.append(np.zeros(n_cap),[read_time(np.median(scores[max(0,i-window_size):i])) for i in range(1,len(scores)+1)][n_cap:]),c='red',label='Speaking Ratio (Median)')
ax.plot(X,np.append(np.zeros(n_cap),[read_time(x) for x in smooth(scores,window_size=window_size,coef=2)][n_cap:]),c='green',label='Speaking Ratio (Exponential Moving Average)')
ax.set_ylim(0,100)
ax.legend()
ax.set_xlabel('Time')
ax.set_title('Speaking Ratio Compared to Scores')
st.pyplot(fig)