import time
import re
import torch
import streamlit as st

from contextlib import contextmanager
from predict import load, get_prediction
from viz import text_highlight, attention_heatmap, network_html
from postprocessing import TitlePostProcessor
from utils import split_tensor_by_words, token_to_words, model_forward

from GenerationArguments import GenerationArguments
from IPython.core.display import HTML

generation_args = GenerationArguments()

@contextmanager
def timer(name) :
    t0 = time.time()
    yield
    st.success(f"[{name}] done in {time.time() - t0:.3f} s")

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")
def main(args):
    st.title("Welcome in text generation website")

    st.info("제목 생성을 위한 본문 내용을 넣어주세요!\n")
    beams_input = st.sidebar.slider('Number of beams search', 1, 5, 3, key='beams')
    layer = st.sidebar.slider('Layer', 0, 5, 5, key='layer')

    
    with timer("load...") :
        tokenizer, model = load(args.model)
    
    model_name = args.use_model
    input_text = st.text_area('Prompt:', height=200)
    if input_text :
        with timer("generate...") : 
            generated_tokens = get_prediction(tokenizer, model, model_name, input_text, beams_input, generation_args)
            title = tokenizer.decode(generated_tokens.squeeze().tolist(), skip_special_tokens=True)
            title = re.sub('</s> |</s>|[CLS] | [SEP]', '', title)

            pcs = TitlePostProcessor()
            title = pcs.post_process(title)
            st.write(f'Titles: {title}')

    if st.button('Attention Highlight'):
        st_cross_attn, enc_input_ids, dec_input_ids = model_forward(model, tokenizer, input_text, title)
        enc_tokens = tokenizer.convert_ids_to_tokens(enc_input_ids[0])
        dec_tokens = tokenizer.convert_ids_to_tokens(dec_input_ids[0])
        
        dec_split = token_to_words(dec_tokens, model_name)
        enc_split = token_to_words(enc_tokens, model_name)

        dec_split_indices = split_tensor_by_words(dec_tokens, model_name)
        enc_split_indices = split_tensor_by_words(enc_tokens, model_name)

        # print(dec_split_indices)
        # breakpoint()

        highlighted_text = text_highlight(st_cross_attn, enc_tokens, model_name)
        st.write(HTML(highlighted_text))
                
        fig = attention_heatmap(st_cross_attn, enc_split, dec_split,
                                enc_split_indices, dec_split_indices, layer)
        st.plotly_chart(fig)

        ## network
        network_html(st_cross_attn, enc_split, dec_split,
                    enc_split_indices, dec_split_indices)

if __name__ == "__main__" :
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../model/checkpoint/baseV1.0_Kobart')
    parser.add_argument('--use_model', type=str, default='kobart', help='kobigbirdbart or etc')
    args = parser.parse_args()

    main(args)