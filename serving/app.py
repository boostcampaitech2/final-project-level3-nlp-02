import time
import re
import torch
import streamlit as st

from contextlib import contextmanager
from predict import load, get_prediction
from viz import text_highlight, attention_heatmap, network_html
from serving.text_processor import PreProcessor, PostProcessor
from utils import split_tensor_by_words, token_to_words, model_forward

from GenerationArguments import GenerationArguments
from IPython.core.display import HTML

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv(dotenv_path='elasticsearch.env')
username = os.getenv("username") 
password = os.getenv('password')

es = Elasticsearch(['https://final-project.es.us-east4.gcp.elastic-cloud.com'], port=9243, http_auth=(username, password))

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
    st.info("좌측에 본문 내용을 넣어주세요!\n")
    
    doc_type = st.sidebar.selectbox('문서 타입을 선택해주세요!', ['해당없음', '기사', '논문', '잡지'])
    
    input_text = st.sidebar.text_area('문서 내용을 입력해주세요!', height=500)
    
    layer = -1
    beams_input = 3 # st.sidebar.slider('Number of beams search', 1, 5, 3, key='beams')

    model_name = args.use_model
    tokenizer, model = load(args.checkpoint, model_name)

    if input_text :
        with timer("generate...") :
            prepcs = PreProcessor()
            processed_input_text = prepcs.pre_process(input_text)
            
            generated_tokens = get_prediction(tokenizer, model, model_name, processed_input_text, beams_input, generation_args)
            title = tokenizer.decode(generated_tokens.squeeze().tolist(), skip_special_tokens=True)
            title = re.sub('</s> |</s>|[CLS] | [SEP]', '', title)

            postpcs = PostProcessor()
            title = postpcs.post_process(title)
            st.write(f'Titles: {title}')

            retrieve_result = es.search(index='summarization_nori', body={'size':5, 'query':{'match':{'text':input_text}}})
            
            col1, col2 = st.columns([1, 3])
            for i in range(5):
                @st.cache(allow_output_mutation=True)
                def button_states():
                    return {"pressed"+str(i): None}

                press_button = col2.button(f"{i+1}번 유사 제목 본문 보기")
                is_pressed = button_states()  # gets our cached dictionary

                if press_button:
                    # any changes need to be performed in place
                    is_pressed.update({"pressed"+str(i): True})

                col1.write(f"유사제목 {i+1}: {retrieve_result['hits']['hits'][i]['_source']['title']}")                
                # press_button = col2.button(f"{i+1}번 유사 제목 본문 보기")
                # if press_button :
                if is_pressed["pressed"+str(i)]:  # saved between sessions
                    col2.write(f"본문내용: {retrieve_result['hits']['hits'][i]['_source']['text']}")
                

    if st.button('Visualization!'):
        
        st_cross_attn, enc_input_ids, dec_input_ids = model_forward(model, tokenizer, processed_input_text, title)
        enc_tokens = tokenizer.convert_ids_to_tokens(enc_input_ids[0])
        dec_tokens = tokenizer.convert_ids_to_tokens(dec_input_ids[0])
        
        dec_split = token_to_words(dec_tokens, model_name)
        enc_split = token_to_words(enc_tokens, model_name)

        dec_split_indices = split_tensor_by_words(dec_tokens, model_name)
        enc_split_indices = split_tensor_by_words(enc_tokens, model_name)

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
    parser.add_argument('--checkpoint', type=str, default='../model/checkpoint/kobigbirdbart_base_ep3_bs8_pre_noam')
    parser.add_argument('--use_model', type=str, default='bigbart', help='bigbart or etc')
    args = parser.parse_args()

    main(args)