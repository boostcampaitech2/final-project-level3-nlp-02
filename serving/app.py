import time
import re
import streamlit as st


from contextlib import contextmanager
from predict import load, get_prediction
from viz import text_highlight, cross_attention
from postprocessing import TitlePostProcessor

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

root_password = 'password'

def main(args):
    st.title("Welcome in text generation website")
    st.balloons()

    st.info("제목 생성을 위한 본문 내용을 넣어주세요!\n")
    beams_input = st.sidebar.slider('Number of beams search', 1, 5, 3, key='beams')

    model_name = args.model
    with timer("load...") :
        tokenizer, model = load(model_name)
    input_text = st.text_area('Prompt:', height=250)

    if input_text :
        with timer("generate...") :
            generated_tokens = get_prediction(tokenizer, model, model_name, input_text, beams_input, generation_args)
            title = tokenizer.decode(generated_tokens.squeeze().tolist(), skip_special_tokens=True)
            title = re.sub('</s> |</s>|[CLS] | [SEP]', '', title)
            
        with timer("post processing...") :
            pcs = TitlePostProcessor()
            title = pcs.post_process(title)
            
            st.write(f'Titles: {title}')
    
    # if st.button('Attention Highlight'):
        highlighted_text = text_highlight(model, tokenizer, input_text, title)
        # col1, col2 = st.columns([2, 2])
        # col1.write(HTML(highlighted_text))
        # col2.write(title)

        st.write(HTML(highlighted_text))

        layer = st.sidebar.slider('Layer', 0, 5, 5, key='layer')
        fig = cross_attention(model, tokenizer, input_text, title, args.use_model, layer)
        st.plotly_chart(fig)

if __name__ == "__main__" :
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../model/checkpoint/baseV1.0_Kobart')
    parser.add_argument('--use_model', type=str, default='kobart', help='kobigbirdbart or etc')
    args = parser.parse_args()

    main(args)