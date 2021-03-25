import cv2
import os
import numpy as np
import random
#import cPickle as pickle
import pickle
import warnings
import argparse
import io
import h5py
from PIL import Image
from gen_vocab import gen_vocab


from translator import prepare

parser = argparse.ArgumentParser(description='Sort-of-CLEVR dataset generator')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--t-subtype', type=int, default=-1,
                    help='Force ternary questions to be of a given type')
parser.add_argument('--finetune-number', type=int, default=0,
                    help='Number of fine-tuning from condition B to include in A')
parser.add_argument('--finetune-split-name', type=str, default='finetune=0',
                    help='Number of fine-tuning from condition B to include in A')
parser.add_argument('--train-size', type=int, default=9800,
                    help='Train size - condition A')
parser.add_argument('--test-size', type=int, default=9800,
                    help='Test size - condition B')
parser.add_argument('--val-size', type=int, default=1000,
                    help='Val size - condition A')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

train_size = 1000
test_size = 10
img_size = 75
size = 5
question_size = 18  ## 2 x (6 for one-hot vector of color), 3 for question type, 3 for question subtype
q_type_idx = 12
sub_q_type_idx = 15
num_objects = 6 
tri_size = 4
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]""" 

# maybe limit  the answer to be only yes or no
# switch the images so that condition B we only have rectangles 
# with "r" " g" and "b" circles with "o" "k" "y" and condition A is switched


nb_questions = 1
dirs = './data'

colors = [
    (0,0,255),##r
    (0,255,0),##g
    (255,0,0),##b
    (0,156,255),##o
    (128,128,128),##k
    (0,255,255)##y
]

try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))

def center_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)        
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center



def draw(img, center, color, shape = 'rectangle'):
    if shape == 'rectangle':
        start = (center[0]-size, center[1]-size)
        end = (center[0]+size, center[1]+size)
        cv2.rectangle(img, start, end, color, -1)

    elif shape == 'circle':
        center_ = (center[0], center[1])
        cv2.circle(img, center_, size, color, -1)

    else: #triangle
        lb = [ center[0]-size ,center[1]-size]
        rb = [ center[0]+size,center[1]-size]
        mt = [ center[0],center[1]+size]
        vertices = np.array([mt, lb, rb], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)
        cv2.fillPoly(img, [pts], color)


def generate_examples(condition):
    objects = []
    img = np.ones((img_size,img_size,3)) * 255

    if condition== 'A': # if condition A then rectnagles are     and circles are o, k, y
        for color_id, color in enumerate(colors):     
            center = center_generate(objects)
            object_id = random.randint(0,1)

            if color_id < 3:
                if object_id == 0: # draw rectangle 
                    draw(img, center, color, shape = 'rectangle')
                    objects.append((color_id,center,'r'))
                else:
                    draw(img, center, color, shape = 'triangle')
                    objects.append((color_id,center,'t'))

            else:
                if object_id == 0:         
                    draw(img, center, color, shape = 'circle')
                    objects.append((color_id,center,'c'))
                else:
                    draw(img, center, color, shape = 'triangle')
                    objects.append((color_id,center,'t'))


    else: # if condition A then rectnagles are r, g, b and circles are o, k, y
        for color_id, color in enumerate(colors):     
            center = center_generate(objects)
            object_id = random.randint(0,1)

            if color_id >= 3:
                if object_id == 0: # draw rectangle 
                    draw(img, center, color, shape = 'rectangle')
                    objects.append((color_id,center,'r'))
                else:
                    draw(img, center, color, shape = 'triangle')
                    objects.append((color_id,center,'t'))

            else:
                if object_id == 0:         
                    draw(img, center, color, shape = 'circle')
                    objects.append((color_id,center,'c'))
                else:
                    draw(img, center, color, shape = 'triangle')
                    objects.append((color_id,center,'t'))

    binary_questions = []
    norel_questions = []
    binary_answers = []
    norel_answers = []
    """Non-relational questions"""
    for _ in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx] = 1

        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        norel_questions.append(question)
        """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y, triangle]"""
        if subtype == 0:
            """query shape->rectangle/circle/triangle"""
            if objects[color][2] == 'r':
                answer = 2
            elif objects[color][2] == 't':
                answer = 10
            else:
                answer = 3

        elif subtype == 1:
            """query horizontal position->yes/no"""
            if objects[color][1][0] < img_size / 2:
                answer = 0
            else:
                answer = 1

        elif subtype == 2:
            """query vertical position->yes/no"""
            if objects[color][1][1] < img_size / 2:
                answer = 0
            else:
                answer = 1
        norel_answers.append(answer)
    
    """Binary Relational questions"""
    for _ in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx+1] = 1
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        binary_questions.append(question)

        if subtype == 0:
            """closest-to->rectangle/circle/triangle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            dist_list[dist_list.index(0)] = 999
            closest = dist_list.index(min(dist_list))
            if objects[closest][2] == 'r':
                answer = 2
            elif objects[closest][2] == 't':
                answer = 10
            else:
                answer = 3
                
        elif subtype == 1:
            """furthest-from->rectangle/circle/triangle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            furthest = dist_list.index(max(dist_list))
            if objects[furthest][2] == 'r':
                answer = 2
            elif objects[furthest][2] == 't':
                answer = 10
            else:
                answer = 3

        elif subtype == 2:
            """count->1~6"""
            my_obj = objects[color][2]
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count +=1 
            answer = count+4

        binary_answers.append(answer)

    #prepare questions 
    binary_questions = prepare(binary_questions)
    norel_questions = prepare(norel_questions)

    dataset = []
    answers = binary_answers + norel_answers
    for i, question in enumerate(binary_questions + norel_questions):
        dataset.append((img, question, answers[i]))
    
    return dataset

def generate_data(condition, 
                  other_condition,
                  size,
                  other_size,
                  name):

    max_question_len = 3
    num_examples = size+other_size 

    path = 'data/' + args.finetune_split_name 
    os.makedirs(path, exist_ok=True) 
    path  += '/' + name

    with h5py.File(path + '_questions.h5', 'w') as dst_questions, h5py.File(path + '_features.h5', 'w') as dst_features:
        features_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
        features_dataset = dst_features.create_dataset('features', (num_examples,), dtype=features_dtype)
        questions_dataset = dst_questions.create_dataset('questions', (num_examples, max_question_len), dtype=np.int64)
        answers_dataset = dst_questions.create_dataset('answers', (num_examples,), dtype=np.int64)
        image_idxs_dataset = dst_questions.create_dataset('image_idxs', (num_examples,), dtype=np.int64)

        i = 0

        num_questions_per_img = nb_questions*2
        # different seeds for train/dev/test
        scenes = []
        while i < num_examples//num_questions_per_img:

            if i < size//num_questions_per_img:
                dataset = generate_examples(condition)
            else:
                print(i)
                dataset = generate_examples(other_condition)

            for j, (scene, question, answer) in enumerate(dataset):

                buffer_ = io.BytesIO()
                image = Image.fromarray(scene.astype(np.uint8))
                image.save(buffer_, format='png')
                buffer_.seek(0)
                features_dataset[i*num_questions_per_img+j]   = np.frombuffer(buffer_.read(), dtype='uint8')
                questions_dataset[i*num_questions_per_img+j]  = question
                answers_dataset[i*num_questions_per_img+j]    = answer
                image_idxs_dataset[i*num_questions_per_img+j] = i*num_questions_per_img+j

            i += 1



print('building train - condition A datasets...')
generate_data('A','B', args.train_size, args.finetune_number, 'train') 

if args.finetune_number != 0:
    val_finetune_number = int (args.val_size * (args.finetune_number / args.train_size ))
else:
    val_finetune_number = 0
    
print('building val - condition A datasets...')
generate_data('A','B',args.val_size, val_finetune_number, 'val') 

print('building test - condition B datasets...')
generate_data('B','A',args.test_size, 0,  'test') 

gen_vocab('data/' + args.finetune_split_name )


