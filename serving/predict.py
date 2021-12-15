import streamlit as st
import torch
from typing import Tuple
from transformers import BartTokenizerFast, BartForConditionalGeneration 

@st.cache(allow_output_mutation=True)
def load(model_name) -> Tuple[BartTokenizerFast, BartForConditionalGeneration] :
    tokenizer = BartTokenizerFast.from_pretrained(
        model_name
    )
    model = BartForConditionalGeneration.from_pretrained(
        model_name,
        output_attentions=True,
    )
    return tokenizer, model

def get_prediction(
    tokenizer:BartTokenizerFast,
    model: BartForConditionalGeneration,
    input_text:str,
    num_beam:int,
    generation_args
    ) -> str:
    with st.spinner('Wait for it...'):
        with torch.no_grad():
            encoder_input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
           
            generated_tokens = model.generate(
                encoder_input_ids["input_ids"],
                attention_mask=encoder_input_ids["attention_mask"],
                num_beams=num_beam,
                **generation_args.__dict__
            )
            return generated_tokens
