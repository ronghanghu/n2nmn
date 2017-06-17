import json
import numpy as np

function2module = {
    'filter_color': '_Filter',
    'filter_material': '_Filter',
    'filter_shape': '_Filter',
    'filter_size': '_Filter',

    'same_color': '_FindSameProperty',
    'same_material': '_FindSameProperty',
    'same_shape': '_FindSameProperty',
    'same_size': '_FindSameProperty',

    'relate': '_Transform',
    'intersect': '_And',
    'union': '_Or',

    'count': '_Count',
    'exist': '_Exist',
    'equal_integer': '_EqualNum',
    'greater_than': '_MoreNum',
    'less_than': '_LessNum',

    'equal_color': '_SameProperty',
    'equal_material': '_SameProperty',
    'equal_shape': '_SameProperty',
    'equal_size': '_SameProperty',

    'query_color': '_Describe',
    'query_material': '_Describe',
    'query_shape': '_Describe',
    'query_size': '_Describe',

    'scene': '_Scene',
    'unique': None
}

def _traversal(program, i):
    funcs = []
    for j in program[i]['inputs']:
        funcs += _traversal(program, j)
    funcs.append(program[i]['function'])
    return funcs

prune_set = {
    'equal_integer', 'greater_than', 'less_than', 'equal_color',
    'equal_material', 'equal_shape', 'equal_size'}
rm_set = {
    'count', 'query_color', 'query_material', 'query_shape', 'query_size'}
def _prune_program(program):
    for f in program:
        if f and f['function'] in prune_set:
            assert(len(f['inputs']) == 2)
            input_f_0 = program[f['inputs'][0]]
            input_f_1 = program[f['inputs'][1]]
            if input_f_0['function'] in rm_set:
                assert(len(input_f_0['inputs']) == 1)
                program[f['inputs'][0]] = None
                f['inputs'][0] = input_f_0['inputs'][0]
            if input_f_1['function'] in rm_set:
                assert(len(input_f_1['inputs']) == 1)
                program[f['inputs'][1]] = None
                f['inputs'][1] = input_f_1['inputs'][0]

    return program

def linearize_program(q):
    program = _prune_program(q['program'])
    # 1. Find root: the root module has no parent
    is_root = np.array([f is not None for f in program])
    for f in program:
        if f is not None:
            is_root[f['inputs']] = False
    if np.sum(is_root) != 1:
        assert(np.sum(is_root) >= 1)
        # remove the roots that are 'scene'
        is_not_scene = np.array([not(f and f['function'] == 'scene') for f in program])
        is_root = np.logical_and(is_root, is_not_scene)
        assert(np.sum(is_root) == 1)

    root = np.argmax(is_root)

    # 2. Post-order traversal to obtain RPN
    funcs = _traversal(program, root)

    # 3. Map modules and fix exps
    q_modules = [function2module[f] for f in funcs]
    q_modules_new = q_modules[:]
    for n_f in range(1, len(q_modules)):
        # replace _Scene + _Filter with _Find
        if q_modules[n_f-1] == '_Scene' and q_modules[n_f] == '_Filter':
            q_modules_new[n_f-1] = None
            q_modules_new[n_f] = '_Find'

    q_modules_new = [m for m in q_modules_new if m is not None]
    return q_modules_new

def add_gt_layout(question_file, save_file):
    with open(question_file) as f:
        questions = json.load(f)['questions']

    for n_q, q in enumerate(questions):
        if (n_q+1) % 1000 == 0:
            print('processing %d / %d' % (n_q+1, len(questions)))
        if 'program' in q:
            q['gt_layout'] = linearize_program(q)

    with open(save_file, 'w') as f:
        json.dump(questions, f)

question_file_trn = '../clevr-dataset/questions/CLEVR_train_questions.json'
save_file_trn = './CLEVR_train_questions_gt_layout.json'
add_gt_layout(question_file_trn, save_file_trn)

question_file_val = '../clevr-dataset/questions/CLEVR_val_questions.json'
save_file_val = './CLEVR_val_questions_gt_layout.json'
add_gt_layout(question_file_val, save_file_val)

question_file_tst = '../clevr-dataset/questions/CLEVR_test_questions.json'
save_file_tst = './CLEVR_test_questions_gt_layout.json'
add_gt_layout(question_file_tst, save_file_tst)
