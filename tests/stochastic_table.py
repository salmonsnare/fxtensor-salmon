import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from fxtensor_salmon import FXTensor
from fractions import Fraction
import itertools
import time

st.title('拡散確率テーブル')
st.sidebar.header("制御")
st.sidebar.markdown("スライダー")


def construct_tensor(num): 
    tensor_m = FXTensor([[], [num, num]], data=np.zeros((num, num)))
    row = np.random.randint(0, num)
    col = np.random.randint(0, num)
    tensor_m.data[row, col] = 1

    tensor_d = FXTensor([[num, num], [num, num]], data=np.zeros((num, num, num, num)))

    for item_i in itertools.product(range(1, num + 1), range(1, num + 1)):
        ii, jj = item_i
        i = ii - 1
        j = jj - 1
        for item_j in itertools.product(range(1, num + 1), range(1, num + 1)):
            kk, ll = item_j
            k = kk - 1
            l = ll - 1
            if item_i == item_j:
                tensor_d.data[i, j, k, l] = 0.8

    for item_i in itertools.product(range(1, num + 1), range(1, num + 1)):
        ii, jj = item_i
        i = ii - 1
        j = jj - 1
        kk = (ii % num) + 1
        ll = (jj % num) + 1
        k = kk - 1
        l = ll - 1
        tensor_d.data[i, j, k, l] = 0.2

    return tensor_m, tensor_d


def create_fig(tensor_result):
    val = np.array([
        [ 
            tensor_result.data[index_i, index_j] for index_i in range(tensor_result.profile[1][0]) 
        ] for index_j in range(tensor_result.profile[1][1])
    ])
    fig, ax = plt.subplots()
    im = ax.imshow(val, cmap='Blues', interpolation='nearest')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(val.shape[1]))
    ax.set_yticks(np.arange(val.shape[0]))
    return fig


def main():
    num = 100
    tensor_m, tensor_d = construct_tensor(num)
    max_step = st.sidebar.slider('最大ステップ数',  min_value=0, max_value=100, step=1, value=10)

    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'running' not in st.session_state:
        st.session_state.running = False

    play = st.sidebar.button('Play')
    if play:
        st.session_state.running = True
        st.session_state.step = 0

    stop = st.sidebar.button('Stop')
    if stop:
        st.session_state.running = False

    tensor_result = tensor_m
    for i in range(st.session_state.step):
        tensor_result = tensor_result.composition(tensor_d)

    st.write("Step {0}".format(st.session_state.step))
    fig = create_fig(tensor_result)
    st.pyplot(fig)

    if st.session_state.running:
        time.sleep(1)
        st.session_state.step += 1
        if st.session_state.step > max_step:
            st.session_state.running = False
        st.rerun()


if __name__ == "__main__":
    main()