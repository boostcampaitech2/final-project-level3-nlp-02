import sys
sys.path.append('..')

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model.models.modeling_kobigbird_bart import EncoderDecoderModel

from model.models.modeling_longformerbart import LongformerBartWithDoctypeForConditionalGeneration

@st.cache(allow_output_mutation=True)
def load(model_name) :
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if "longformerbart" in model_name:
        model = LongformerBartWithDoctypeForConditionalGeneration.from_pretrained(model_name)
    elif "kobigbirdbart" in model_name:
        tokenizer = AutoTokenizer.from_pretrained('monologg/kobigbird-bert-base')
        model = EncoderDecoderModel.from_pretrained(model_name, output_attentions=True)
        model.encoder.encoder.layer = model.encoder.encoder.layer[:model.config.encoder.encoder_layers]
        model.encoder.config.output_attentions = True
        model.decoder.config.output_attentions = True
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        output_attentions=True,
    )
    return tokenizer, model

def get_prediction(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    model_name: str,
    input_text: str,
    num_beam: int,
    generation_args
    ) -> str:
    with st.spinner('Wait for it...'):
        with torch.no_grad():
            input_ids = tokenizer(input_text, add_special_tokens=True)
            if "bigbart" not in model_name :
                input_ids = [tokenizer.bos_token_id] + input_ids['input_ids'][:-2] + [tokenizer.eos_token_id]
           
            generated_tokens = model.generate(
            torch.tensor([input_ids]), num_beams=num_beam, **generation_args.__dict__)

            return generated_tokens
