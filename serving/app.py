import time
import re
import streamlit as st

from contextlib import contextmanager
from predict import load, get_prediction
from viz import text_highlight, cross_attention

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

def main():
    st.title("Welcome in text generation website")
    st.balloons()

    st.info("제목 생성을 위한 본문 내용을 넣어주세요!\n")
    
    beams_input = st.sidebar.slider('Number of beams search', 1, 5, 3, key='beams')

    model_name = '/opt/ml/final-project-level3-nlp-02/model/baseV1.0_Kobart'
    with timer("load...") :
        tokenizer, model = load(model_name)
    input_text = st.text_area('Prompt:', height=400)

    if input_text :
        with timer("generate...") :
            generated_tokens = get_prediction(tokenizer, model, input_text, beams_input, generation_args)
            titles = tokenizer.decode(generated_tokens.squeeze().tolist(), skip_special_tokens=True)
            titles = re.sub('</s> |</s>', '', titles)
            st.write(f'Titles: {titles}')
    
    if st.button('Attention Highlight'):
        highlighted_text = text_highlight(model, tokenizer, input_text, generated_tokens)
        col1, col2 = st.columns([2, 2])
        
    
        col1.write(HTML(highlighted_text))
        col2.write(titles)

        fig = cross_attention(model, tokenizer, input_text, generated_tokens)
        st.plotly_chart(fig)
main()