import json

def gen_vocab(path):
    token2idx = {
        "<NULL>": 0,
        "<START>":1,
        "<END>":2,
        "red": 3,
        "green": 4,
        "blue": 5,
        "orange": 6,
        "gray": 7,
        "yellow": 8,
        "which": 9,
        "closest": 10,
        "furthest": 11,
        "shape": 12,
        "count": 13,
        "left": 14,
        "up": 15
        }
    txt2idx = {
        'red':1,
        'green':2,
        'blue':3,
        'orange':4,
        'gray':5,
        'yellow':6,
        'which':7,
        'closest':8,
        'furthest':9,
        'shape':10,
        'count':11,
        'left':12,
        'up':13,
        }
    program_token_to_idx =  {
        "<NULL>": 0,
        "<START>":1,
        "<END>":2,
      }

    idx2token = {value:key for key, value in token2idx.items()}
    answer = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6', 'triangle']
    answer2idx = {w:i for i, w in enumerate(answer)}
    idx2answer = {i:w for i, w in enumerate(answer)}

    dic = {
    	'question_token_to_idx': token2idx,
    	'answer_token_to_idx': answer2idx,
    	"text_token_to_idx":txt2idx,
        "program_token_to_idx":program_token_to_idx
    }

    with open(path + '/vocab.json', 'w') as dst:
        json.dump(dic, dst, indent=2)