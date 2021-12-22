import sys

from torch.functional import _return_output
sys.path.append('..')

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model.models.modeling_kobigbird_bart import EncoderDecoderModel
from model.models.modeling_longformerbart import LongformerBartWithDoctypeForConditionalGeneration

@st.cache(allow_output_mutation=True)
def load(checkpoint, model_name) :
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    if "longformerbart" == model_name:
        model = LongformerBartWithDoctypeForConditionalGeneration.from_pretrained(checkpoint)
    elif "bigbart" == model_name:
        print('encoder decoder')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = EncoderDecoderModel.from_pretrained(checkpoint, output_attentions=True)
        model.encoder.config.output_attentions = True
        model.decoder.config.output_attentions = True
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
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
                input_ids = [tokenizer.bos_token_id] + input_ids['input_ids'][:-1]
            else :
                input_ids = input_ids['input_ids']
            
            generated_tokens = model.generate(
                torch.tensor([input_ids]),
                num_beams=num_beam,
                **generation_args.__dict__)

            # generated_tokens = model.generate(
            #     torch.tensor([input_ids]),
            #     num_beams=num_beam)

            return generated_tokens
