import re
import os
import time
import streamlit as st

from transformers import HfArgumentParser
from contextlib import contextmanager
from predict import load, get_prediction
from viz import text_highlight, attention_heatmap, network_html
from utils import split_tensor_by_words, token_to_words, model_forward
from serving.text_processor import PreProcessor, PostProcessor

import sys
sys.path.append('..')
from model.args import (
    GenerationArguments,
    DataTrainingArguments
)

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from IPython.core.display import HTML

load_dotenv(dotenv_path='elasticsearch.env')
username = os.getenv("username") 
password = os.getenv('password')

es = Elasticsearch(['https://final-project.es.us-east4.gcp.elastic-cloud.com'], port=9243, http_auth=(username, password))

data_args, gen_args = DataTrainingArguments, GenerationArguments
@contextmanager
def timer(name) :
    t0 = time.time()
    yield
    st.success(f"[{name}] done in {time.time() - t0:.3f} s")

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")
def main(args):
    # parser = HfArgumentParser(
    #     (DataTrainingArguments, GenerationArguments)
    # )
    # data_args, gen_args = parser.parse_args_into_dataclasses()
    # data_args.max_source_length = 2048

    st.title("Welcome in text generation website")
    st.info("좌측에 본문 내용을 넣어주세요!\n")
    
    doc_type = st.sidebar.selectbox('문서 타입을 선택해주세요!', ['해당없음', '기사', '논문', '잡지'])
        
    input_text = st.sidebar.text_area('문서 내용을 입력해주세요!', height=500)
    layer = -1
    beams_input = 3 # st.sidebar.slider('Number of beams search', 1, 5, 3, key='beams')

    model_name = args.use_model
    tokenizer, model = load(args.checkpoint, model_name)
   
    if doc_type != '해당없음' and model_name == 'bigbart_tapt':
        data_args.use_doc_type_ids = True

    if input_text :
        with timer("generate...") :
            prepcs = PreProcessor()
            processed_input_text = prepcs.pre_process(input_text)
            
            generated_tokens = get_prediction(tokenizer, model, doc_type, processed_input_text, beams_input, data_args, gen_args)
            title = tokenizer.decode(generated_tokens.squeeze().tolist(), skip_special_tokens=True)
            title = re.sub('</s> |</s>|[CLS] | [SEP]', '', title)

            postpcs = PostProcessor()
            title = postpcs.post_process(title)
            st.write(f'Titles: {title}')

            retrieve_result = es.search(index='summarization_unique', body={'size':5, 'query':{'match':{'text':input_text}}})
            for i in range(3):
                ret_title = retrieve_result['hits']['hits'][i]['_source']['title']
                ret_title = ret_title.replace('`','\'')
                st.write(f"유사제목 {i+1}: {ret_title}")

                button = st.button(f"{i+1} 유사제목 본문보기!")
                if button :
                    ret_text = retrieve_result['hits']['hits'][i]['_source']['text']
                    st.write(f"본문내용 {i+1}: {ret_text}")
                    

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
    # parser.add_argument('--checkpoint', type=str, default='../model/checkpoint/kobigbirdbart_base_ep3_bs8_pre_noam')
    parser.add_argument('--checkpoint', type=str, default='metamong1/bigbart_tapt_ep3_bs16_pre_noam')
    # parser.add_argument('--use_model', type=str, default='bigbart', help='bart, bigbart, bigbart_tapt or etc..')
    parser.add_argument('--use_model', type=str, default='bigbart_tapt', help='bart, bigbart, bigbart_tapt or etc..')
    args = parser.parse_args()

    main(args)