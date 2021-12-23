import torch
import numpy as np
import uuid
from collections import Counter
from pyvis.network import Network
import plotly.graph_objects as go
import streamlit.components.v1 as components
from utils import position
from typing import List

def rgb_to_hex(r, g, b):
    r, g, b = int(r), int(g), int(b)
    return '#' + hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)

def highlighter(
        color: str,
        word: str
    ) -> str :
    word = '<span style="background-color:' +color+ '">' +word+ '</span>'
    return word

def text_highlight(st_cross_attn, encoder_tokens, model_type) :
    layer_mat = st_cross_attn.detach()
    last_h_layer_mat = torch.mean(layer_mat, 1)[-1] ## mean by head side, last layer
    enc_mat = torch.mean(last_h_layer_mat, 0) ## mean by decoder id side

    enc_mat-= enc_mat.min()
    enc_mat /= enc_mat.max()
    
    colors = [rgb_to_hex(255, 255, 255*(1-attn_s)) for attn_s in enc_mat.numpy()]
    if 'bigbart' in model_type :
        encoder_tokens = ['▁'+word if '##' not in word else word.replace('##','') for word in encoder_tokens ]
    higlighted_text = ''.join([highlighter(colors[i], word) for i, word in enumerate(encoder_tokens)])
    higlighted_text = higlighted_text.replace('▁',' ')
    return higlighted_text

def attention_heatmap(
        st_cross_attn: torch.Tensor, # (layer, head, dec_len, enc_len)
        enc_split: List[str],
        dec_split: List[str],
        enc_split_indices: List[int],
        dec_split_indices: List[int],
        layer: int
    ) -> go.Figure :

    last_h_layer_mat = torch.mean(st_cross_attn.detach(), 1)[layer] # (dec_len, enc_len)

    splited_by_spaces = torch.split(last_h_layer_mat, dec_split_indices, dim=0)
    merging_tensor = []
    for split_tensor in splited_by_spaces :
        merging_tensor.append(torch.mean(split_tensor, 0))
    merged_attn = torch.stack(merging_tensor, dim=0)

    splited_by_spaces = torch.split(merged_attn, enc_split_indices, dim=1)
    merging_tensor = []
    for split_tensor in splited_by_spaces :
        merging_tensor.append(torch.mean(split_tensor, 1))
    merged_attn = torch.stack(merging_tensor, dim=1)

    go_fig = go.Figure(go.Heatmap(
                    z=merged_attn,
                    x=enc_split,
                    y=dec_split,
                    colorscale='Reds',
                    hoverongaps=False))
    go_fig.update_layout(
        autosize=False,
        width=1200,
        height=400,
        yaxis_autorange="reversed",
        margin=dict(l=10, r=20, t=20, b=20)
    )
    go_fig.update_xaxes(tickangle = 45)
    return go_fig
    
def update_mapping(
        uuid_mapping:dict,
        node_list: List[str]
    ) -> List[int]:
    unique_node_list = []
    for node in node_list:
        # Create unique ID for node
        unique_name = uuid.uuid4()
        unique_node_list.append(unique_name.int)
        uuid_mapping[unique_name]=node
    return unique_node_list
    

def transparent_by_attn(
        attn_matrix: torch.Tensor,
        dec_idx: int,
        enc_idx: int,
    ) -> str:
    ori_hex = "#8080C0"
    attn_matrix = (attn_matrix - attn_matrix.min())/(attn_matrix.max()-attn_matrix.min())
    if attn_matrix[dec_idx][enc_idx] == 1:
        return ori_hex
    else :
        ## 투명도 설정
        transparent = '{0:02d}'.format(int(round(attn_matrix[dec_idx][enc_idx], 2) * 50) + 50) 
        ori_hex = "#8080C0" + transparent
        return ori_hex

def network_html(
        st_cross_attn: torch.Tensor, # (layer, head, dec_len, enc_len)
        enc_split: List[str],
        dec_split: List[str],
        enc_split_indices: List[int],
        dec_split_indices: List[int],
    ) -> None :
    uuid_net = Network("600px", "1000px")
    uuid_mapping = {}
    
    enc_nodes = update_mapping(uuid_mapping, enc_split)
    dec_nodes = update_mapping(uuid_mapping, dec_split)

    last_h_layer_mat = torch.mean(st_cross_attn.detach(), 1)[-1] # (dec_len, enc_len)
    splited_by_spaces = torch.split(last_h_layer_mat, dec_split_indices, dim=0)
    merging_tensor = []
    for split_tensor in splited_by_spaces :
        merging_tensor.append(torch.mean(split_tensor, 0))
    merged_attn = torch.stack(merging_tensor, dim=0)

    splited_by_spaces = torch.split(merged_attn, enc_split_indices, dim=1)
    merging_tensor = []
    for split_tensor in splited_by_spaces :
        merging_tensor.append(torch.mean(split_tensor, 1))
    merged_attn = torch.stack(merging_tensor, dim=1)

    attn_matrix = merged_attn.type(torch.half).detach().numpy().astype(np.float) # (dec_len, enc_len)

    attn_max = attn_matrix.max()
    attn_min = attn_matrix.min()

    x_pos_list, y_pos_list = position(len(dec_split))

    threshold = np.mean(attn_matrix)
    for dec_idx, dec_token in enumerate(dec_nodes) :
        uuid_net.add_node(dec_token,
                          label=dec_split[dec_idx],
                          color='#CD6155',
                          size=20,
                          x=x_pos_list[dec_idx],
                          y=y_pos_list[dec_idx])
        for enc_idx, enc_token in enumerate(enc_nodes) :
            if threshold < attn_matrix[dec_idx][enc_idx] : 
                tp_color = transparent_by_attn(attn_matrix, dec_idx, enc_idx)
                norm_attn = 5*(attn_matrix[dec_idx][enc_idx]-attn_min)/(attn_max-attn_min)
                uuid_net.add_node(enc_token, label=enc_split[enc_idx], color=tp_color, size=20) ##  투명도를 attn score          
                uuid_net.add_edge(dec_token, enc_token, width=norm_attn)

    for n in uuid_net.nodes:
        if n['id'] in dec_nodes:
            n.update({'physics': False})
        elif n['id'] in enc_nodes:
            n.update({'physics': True})

    edges_counter = Counter([edges['to'] for edges in uuid_net.get_edges()])
    nodes_over2 = [ k for k,v in edges_counter.items() if v >= len(dec_split)/2]
    nodes_over2 += list(set([edges['from'] for edges in uuid_net.get_edges()]))

    for n in uuid_net.nodes:
        if n['id'] in nodes_over2 :
            n.update({'physics': False})
    # uuid_net.show_buttons(filter_=['physics', 'nodes', 'edges'])
    html_name = 'uuid_example1.html'
    uuid_net.write_html(html_name)

    HtmlFile = open(html_name, 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height = 900, width=900)