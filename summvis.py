import argparse
import json
import operator
import os
import re
from pathlib import Path

import spacy
import streamlit as st
from robustnessgym import Dataset, Identifier
from robustnessgym import Spacy
from spacy.tokens import Doc

from align import NGramAligner, BertscoreAligner, StaticEmbeddingAligner
from components import MainView
from preprocessing import NGramAlignerCap, StaticEmbeddingAlignerCap, \
    BertscoreAlignerCap
from preprocessing import _spacy_decode, _spacy_encode
from utils import clean_text

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MIN_SEMANTIC_SIM_THRESHOLD = 0.1
MAX_SEMANTIC_SIM_TOP_K = 10

Doc.set_extension("name", default=None, force=True)
Doc.set_extension("column", default=None, force=True)


class Instance():
    def __init__(self, id_, document, reference, preds, data=None):
        self.id = id_
        self.document = document
        self.reference = reference
        self.preds = preds
        self.data = data


@st.cache(allow_output_mutation=True)
def load_from_index(filename, index):
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line.strip())


@st.cache(allow_output_mutation=True)
def load_dataset(path: str):
    if path.endswith('.jsonl'):
        return Dataset.from_jsonl(path)
    try:
        return Dataset.load_from_disk(path)
        # return Dataset.
    except NotADirectoryError:
        return Dataset.from_jsonl(path)


@st.cache(allow_output_mutation=True)
def get_model(model_name_or_path):
    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    #     model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    # except:
    #     print("Cannot load Tokenizer / Model !")
    #     raise KeyboardInterrupt
    tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-summarization')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    return tokenizer, model


def retrieve(dataset, index, filename=None):
    if index >= len(dataset):
        st.error(f"Index {index} exceeds dataset length.")

    eval_dataset = None
    if filename:
        # TODO Handle this through dedicated fields
        if "cnn_dailymail" in filename:
            eval_dataset = "cnndm"
        elif "xsum" in filename:
            eval_dataset = "xsum"

    data = dataset[index] 
    # "Spacy(lang=en, pipeline=['tok2vec', 'tagger', 'sentencizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'],
    #       columns=['preprocessed_summary:pegasus-newsroom'])" :
    #                                                           {'arr': array([~]),
    #                                                            'shape': [58, 12],
    #                                                            'words': ['(', 'CNN', ')', '--', 'Singer', '-', 'songwriter', 'David' ~],
    #                                                            }
    # 이렇게 Spacy가 여러 개 +
    #               "BertscoreAlignerCap(threshold=0.1, top_k=1-, columns=['preprocessed_document', 'preprocessed_summary:reference']))" :
    #                   '[{"0": [[22, 0.7785049676895142], [23, 0.45159032940864563], ...]
    #                      "1": [[23, 0.6654687523841858], [22, 0.319236159324646], ...]}]
    #                      ... "28"까지 있음' 이것도 여러 개
    id_ = data.get('id', '') # '0044e296ecfe3ba57a351ad2a36d034491e878ce'

    try:
        document = rg_spacy.decode(
            data[rg_spacy.identifier(columns=['preprocessed_document'])] # 원래 rg_spacy에 columns가 없었는데 여기서 추가됨
        ) # 쪼개져 있던 게 decode으로 합쳐짐
    except KeyError:
        if not is_lg:
            st.error("'en_core_web_lg model' is required unless loading from cached file."
                     "To install: 'python -m spacy download en_core_web_lg'")
        try:
            text = data['document']
        except KeyError:
            text = data['article']
        if not text:
            st.error("Document is blank")
            return
        document = nlp(text if args.no_clean else clean_text(text))
    document._.name = "Document" # 원래 None
    document._.column = "document"

    try:
        reference = rg_spacy.decode(
            data[rg_spacy.identifier(columns=['preprocessed_summary:reference'])]
        )
    except KeyError:
        if not is_lg:
            st.error("'en_core_web_lg model' is required unless loading from cached file."
                     "To install: 'python -m spacy download en_core_web_lg'")
        try:
            text = data['summary'] if 'summary' in data else data['summary:reference']
        except KeyError:
            text = data.get('highlights')
        if text:
            reference = nlp(text if args.no_clean else clean_text(text))
        else:
            reference = None
    if reference is not None:
        reference._.name = "Reference"
        reference._.column = "summary:reference"

    model_names = set()
    for k in data: # 'id', 'index', 'summary:bart-cnndm', 'bart-cnndm:rouge', 'bart-cnndm:bertscore-bert', 'bart-cnndm:bertscore-deberta',
        #                           'summary:bart-xsum', 'bart-xsum:rouge', 'bart-xsum:bertscore-bert', 'bart-xsum:bertscore-deberta',
        #                           'summary:pegasus-cnndm', ... , 'preprocessed_summary:bart-cnndm', 'preprocessed_summary:bart-xsum', ...
        #                           "Spacy(lang=en, pipeline=['tok2vec', 'tagger', 'sentencizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'], columns=['preprocessed_summary:bart-cnndm'])",
        #                           "BertscoreAlignerCap(~)", "StaticEmbeddingAlignerCap(~), "NGramAlignerCap(~)",
        #                           'document', 'summary:reference', 'preprocessed_document', 'preprocessed_summary:reference'
        m = re.match('(preprocessed_)?summary:(?P<model>.*)', k)
        if m:
            model_name = m.group('model') # 'bart-cnndm', 'bart-xsum', 'pegasus-cnndm', 'pegasus-multinews', 'pegasus-newsroom', 'pegasus-xsum', 
            if model_name != 'reference': # 'summary:reference' -> model_name == 'reference', 'preprocessed_summary:reference' -> model_name == 'reference'
                model_names.add(model_name)

    preds = []
    for model_name in model_names: # {'pegasus-multinews', 'bart-cnndm', 'pegasus-xsum', 'bart-xsum', 'pegasus-newsroom', 'pegasus-cnndm'}
        try:
            pred = rg_spacy.decode(
                data[rg_spacy.identifier(columns=[f"preprocessed_summary:{model_name}"])]
            )
        except KeyError:
            if not is_lg:
                st.error("'en_core_web_lg model' is required unless loading from cached file."
                         "To install: 'python -m spacy download en_core_web_lg'")
            text = data[f"summary:{model_name}"]
            pred = nlp(text if args.no_clean else clean_text(text))

        parts = model_name.split("-") # 'pegasus-multinews' -> ['pegasus', 'multinews']
        primary_sort = 0
        if len(parts) == 2:
            model, train_dataset = parts
            if train_dataset == eval_dataset:
                formatted_model_name = model.upper()
            else:
                formatted_model_name = f"{model.upper()} ({train_dataset.upper()}-trained)" # 'PEGASUS (MULTINEWS-trained)'
                if train_dataset in ["xsum", "cnndm"]:
                    primary_sort = 1
                else:
                    primary_sort = 2
        else:
            formatted_model_name = model_name.upper()
        pred._.name = formatted_model_name
        pred._.column = f"summary:{model_name}"
        preds.append(
            ((primary_sort, formatted_model_name), pred)
        )

    preds = [pred for _, pred in sorted(preds)]

    return Instance(
        id_=id_,
        document=document,
        reference=reference,
        preds=preds,
        data=data,
    )


def filter_alignment(alignment, threshold, top_k):
    filtered_alignment = {}
    for k, v in alignment.items():
        filtered_matches = [(match_idx, score) for match_idx, score in v if score >= threshold]
        if filtered_matches:
            filtered_alignment[k] = sorted(filtered_matches, key=operator.itemgetter(1), reverse=True)[:top_k]
    return filtered_alignment


def select_comparison(example):
    all_summaries = []

    if example.reference:
        all_summaries.append(example.reference)
    if example.preds:
        all_summaries.extend(example.preds)

    from_documents = [example.document]
    if example.reference:
        from_documents.append(example.reference)
    document_names = [document._.name for document in from_documents] # ['Document', 'Reference']
    select_document_name = sidebar_placeholder_from.selectbox(
        label="Comparison FROM:",
        options=document_names
    ) # 'Document'
    document_index = document_names.index(select_document_name) # 0
    selected_document = from_documents[document_index]

    remaining_summaries = [summary for summary in all_summaries if
                           summary._.name != selected_document._.name]
    remaining_summary_names = [summary._.name for summary in remaining_summaries] # ['Reference', 'BART', 'PEGASUS', 'BART (XSUM-trained)', 'PEGASUS (XSUM-trained)', 'PEGASUS (MULTINEWS-trained)', 'PEGASUS (NEWSROOM-trained)']

    selected_summary_names = sidebar_placeholder_to.multiselect(
        'Comparison TO:',
        remaining_summary_names,
        remaining_summary_names
    ) # ['Reference', 'BART', 'PEGASUS', 'BART (XSUM-trained)', 'PEGASUS (XSUM-trained)', 'PEGASUS (MULTINEWS-trained)', 'PEGASUS (NEWSROOM-trained)']
    selected_summaries = []
    for summary_name in selected_summary_names:
        summary_index = remaining_summary_names.index(summary_name)
        selected_summaries.append(remaining_summaries[summary_index])
    return selected_document, selected_summaries


def show_main(example, tokenizer):
    # Get user input

    # semantic_sim_type = st.sidebar.radio(
    #     "Semantic similarity type:",
    #     ["Contextual embedding", "Static embedding"]
    # )
    # semantic_sim_threshold = st.sidebar.slider(
    #     "Semantic similarity threshold:",
    #     min_value=MIN_SEMANTIC_SIM_THRESHOLD,
    #     max_value=1.0,
    #     step=0.1,
    #     value=0.2,
    # )
    # semantic_sim_top_k = st.sidebar.slider(
    #     "Semantic similarity top-k:",
    #     min_value=1,
    #     max_value=MAX_SEMANTIC_SIM_TOP_K,
    #     step=1,
    #     value=10,
    # )

    # document, summaries = select_comparison(example)
    layout = st.sidebar.radio("Layout:", ["Vertical", "Horizontal"]).lower() # 'vertical'
    scroll = True
    gray_out_stopwords = st.sidebar.checkbox(label="Gray out stopwords", value=True) # True

    # Gather data
    # try:
    #     lexical_alignments = [
    #         NGramAlignerCap.decode(
    #             example.data[
    #                 Identifier(NGramAlignerCap.__name__)(
    #                     columns=[
    #                         f'preprocessed_{document._.column}',
    #                         f'preprocessed_{summary._.column}',
    #                     ]
    #                 )
    #             ])[0]
    #         for summary in summaries
    #     ]
    #     lexical_alignments = [
    #         {k: [(pair[0], int(pair[1])) for pair in v]
    #          for k, v in d.items()}
    #         for d in lexical_alignments
    #     ]
    # except KeyError:
        # lexical_alignments = NGramAligner().align(document, summaries)
    lexical_alignments = NGramAligner().align(example["source"], example["title"])

    # if semantic_sim_type == "Static embedding":
    #     try:
    #         semantic_alignments = [
    #             StaticEmbeddingAlignerCap.decode(
    #                 example.data[
    #                     Identifier(StaticEmbeddingAlignerCap.__name__)(
    #                         threshold=MIN_SEMANTIC_SIM_THRESHOLD,
    #                         top_k=MAX_SEMANTIC_SIM_TOP_K,
    #                         columns=[
    #                             f'preprocessed_{document._.column}',
    #                             f'preprocessed_{summary._.column}',
    #                         ]
    #                     )
    #                 ])[0]
    #             for summary in summaries
    #         ]
    #     except KeyError:
    #         semantic_alignments = StaticEmbeddingAligner(
    #             semantic_sim_threshold,
    #             semantic_sim_top_k).align(
    #             document,
    #             summaries
    #         )
    #     else:
    #         semantic_alignments = [
    #             filter_alignment(alignment, semantic_sim_threshold, semantic_sim_top_k)
    #             for alignment in semantic_alignments
    #         ]
    # else:
    #     try:
    #         semantic_alignments = [
    #             BertscoreAlignerCap.decode(
    #                 example.data[
    #                     Identifier(BertscoreAlignerCap.__name__)(
    #                         threshold=MIN_SEMANTIC_SIM_THRESHOLD,
    #                         top_k=MAX_SEMANTIC_SIM_TOP_K,
    #                         columns=[
    #                             f'preprocessed_{document._.column}',
    #                             f'preprocessed_{summary._.column}',
    #                         ]
    #                     )
    #                 ])[0]
    #             for summary in summaries
    #         ]
    #     except KeyError:
    #         semantic_alignments = BertscoreAligner(semantic_sim_threshold,
    #                                                semantic_sim_top_k).align(document,
    #                                                                          summaries)
    #     else:
    #         semantic_alignments = [
    #             filter_alignment(alignment, semantic_sim_threshold, semantic_sim_top_k)
    #             for alignment in semantic_alignments
    #         ]
    
    # MainView(
    #     document,
    #     summaries,
    #     semantic_alignments,
    #     lexical_alignments,
    #     layout,
    #     scroll,
    #     gray_out_stopwords,
    # ).show(height=720)
    MainView(
        document = example["source"],
        summaries = example["title"],
        # semantic_alignments,
        lexical_alignments = lexical_alignments,
        layout=layout,
        scroll=scroll,
        gray_out_stopwords=gray_out_stopwords,
        tokenizer=tokenizer,
    ).show(height=720)


if __name__ == "__main__":

    st.set_page_config(layout="wide")

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--file', type=str)
    parser.add_argument('--no_clean', action='store_true', default=False,
                        help="Do not clean text (remove extraneous spaces, newlines).")
    # parser.add_argument('--model_name_or_path', type=str, default='gogamza/kobart-summarization')
    parser.add_argument('--model_name_or_path', type=str, default='baseV1.0_Kobart_ep3_0.7')
    args = parser.parse_args()
    tokenizer, model = get_model(args.model_name_or_path)

    col1, col2 = st.beta_columns((1, 3)) # beta_columns -> 화면 레이아웃 나누는 것
    # filename = col1.selectbox(label="File:", options=files, index=file_index)
    doc_type = col1.selectbox(label="문서 종류: ",
                            options=('논문', '뉴스기사', '사설잡지'))
    source = col2.text_area(label="문서 내용: ")

    raw_input_ids = tokenizer(source, max_length=1024, truncation=True)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids['input_ids'][:-2] + [tokenizer.eos_token_id]
    output_ids = model.generate(torch.tensor([input_ids]), num_beams=3, num_return_sequences=1)

    title = tokenizer.decode(output_ids.squeeze().tolist(), skip_special_tokens=True)
    print(title)
    # if len(output_ids.shape) == 1  or output_ids.shape[0] == 1:
    #         title = tokenizer.decode(output_ids.squeeze().tolist(), skip_special_tokens=True)
    #         st.write(title)
    #     else:
    #         titles = tokenizer.batch_decode(output_ids.squeeze().tolist(), skip_special_tokens=True)
    #         for idx, title in enumerate(titles):
    #             st.write(idx, title)
    
    # raise KeyboardInterrupt
    
    # dataset_size = len(dataset) # 10
    

    # query = col2.number_input(f"Index (Size: {dataset_size}):", value=0, min_value=0, max_value=dataset_size - 1)

    sidebar_placeholder_from = st.sidebar.empty()
    sidebar_placeholder_to = st.sidebar.empty()

    if title is not None:
        # example = retrieve(dataset, query, filename) # dataset, index, filename
        # if example:
        #     show_main(example)
        example = {"source": source, "title": [[title]]}
        show_main(example, tokenizer)
