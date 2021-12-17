
def split_tensor_by_words(text_tokens, model_type) :
    i = 0
    split_words_indices = []
    if model_type != 'bigbart' :    
        for token in text_tokens :
            if '▁' in token:
                split_words_indices.append(i)
                i = 1
            else : 
                i += 1
        split_words_indices.append(i)
    else :
        for token in text_tokens :
            if '##' in token :
                i += 1
            else :
                split_words_indices.append(i)
                i = 1
    split_words_indices = split_words_indices[1:]
    return split_words_indices



def token_to_words(text_tokens, model_type) :
    if model_type != 'bigbart' :    
        join_text = ''.join(text_tokens).replace('▁', ' ')    
        space_text = join_text.split(' ')[1:]
    else :
        join_text = ' '.join(text_tokens).replace(' ##', '')
        space_text = join_text.split(' ')
    return space_text
    
