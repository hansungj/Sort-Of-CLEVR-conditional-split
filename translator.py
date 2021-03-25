import cv2
import numpy as np
def translate_old(dataset):
    img, (rel_questions, rel_answers), (norel_questions, norel_answers) = dataset
    colors = ['red ', 'green ', 'blue ', 'orange ', 'gray ', 'yellow ']
    answer_sheet = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6', 'triangle']
    questions = rel_questions + norel_questions
    answers = rel_answers + norel_answers

    print(questions)
    print(answers)


    for question, answer in zip(questions,answers):
        query = ''
        query += colors[question.tolist()[0:6].index(1)]

        if question[12] == 1:
            if question[15] == 1:
                query = 'which shape ' + query
            if question[16] == 1:
                query += 'shape left'
            if question[17] == 1:
                query += 'shape up?'
        if question[13] == 1:
            if question[15] == 1:
                query += 'closest shape?'
            if question[16] == 1:
                query += 'furthest shape?'
            if question[17] == 1:
                query += "shape count?"

        ans = answer_sheet[answer]
        print(query,'==>', ans)
    #cv2.imwrite('sample.jpg',(img*255).astype(np.int32))
    cv2.imshow('img',cv2.resize(img,(512,512)))
    cv2.waitKey(0)

def translate(dataset):
    img, (rel_questions, rel_answers), (norel_questions, norel_answers) = dataset
    colors = ['red ', 'green ', 'blue ', 'orange ', 'gray ', 'yellow ']
    answer_sheet = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6', 'triangle']
    questions = rel_questions + norel_questions
    answers = rel_answers + norel_answers

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
    
    idx2token = {value:key for key, value in token2idx.items()}

    for question, answer in zip(questions,answers):
        query = ' '.join([idx2token[int(index)] for index in question])
        ans = answer_sheet[answer]
        print(query,'==>', ans)
    #cv2.imwrite('sample.jpg',(img*255).astype(np.int32))
    cv2.imshow('img',cv2.resize(img,(512,512)))
    cv2.waitKey(0)

def prepare(questions):

    qst = []
    for i, question in enumerate(questions):
        onehot = np.zeros(3, np.int32)
        # onehot[0] = 1
        # onehot[-1] =2
        color = question[0:6].tolist().index(1) + 3

        if question[12] == 1:
            if question[15] == 1: #'which shape color '
                onehot[0] = 9
                onehot[1] = 12
                onehot[2] = color
            if question[16] == 1: #'color shape left '
                onehot[0] = color
                onehot[1] = 12
                onehot[2] = 14
            if question[17] == 1: #'color shape up?'
                onehot[0] = color
                onehot[1] = 10
                onehot[2] = 15

        if question[13] == 1:
            if question[15] == 1: #'closest shape?'
                onehot[0] = color
                onehot[1] = 10
                onehot[2] = 12
            if question[16] == 1: #'furthest shape?'
                onehot[0] = color
                onehot[1] = 11
                onehot[2] = 12
            if question[17] == 1: #"shape count?"
                onehot[0] = color
                onehot[1] = 12
                onehot[2] = 13
        qst.append(onehot)

    return qst


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
idx2token = {value:key for key, value in token2idx.items()}