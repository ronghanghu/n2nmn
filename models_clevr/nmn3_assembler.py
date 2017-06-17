from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_fold as td
from tensorflow import convert_to_tensor as to_T

# the number of attention input to each module
_module_input_num = {
    '_Scene': 0,
    '_Find': 0,
    '_Filter': 1,
    '_FindSameProperty': 1,
    '_Transform': 1,
    '_And': 2,
    '_Or': 2,
    '_Count': 1,
    '_Exist': 1,
    '_EqualNum': 2,
    '_MoreNum': 2,
    '_LessNum': 2,
    '_SameProperty': 2,
    '_Describe': 1}

# output type of each module
_module_output_type = {
    '_Scene': 'att',
    '_Find': 'att',
    '_Filter': 'att',
    '_FindSameProperty': 'att',
    '_Transform': 'att',
    '_And': 'att',
    '_Or': 'att',
    '_Count': 'ans',
    '_Exist': 'ans',
    '_EqualNum': 'ans',
    '_MoreNum': 'ans',
    '_LessNum': 'ans',
    '_SameProperty': 'ans',
    '_Describe': 'ans'}

INVALID_EXPR = 'INVALID_EXPR'
# decoding validity: maintaining a state x of [#att, #ans, T_remain]
# when T_remain is T_decoder when decoding the first module token
# a token s can be predicted iff all(<x, w_s> - b_s >= 0)
# the validity token list is
#       XW - b >= 0
# the state transition matrix is P, so the state update is X += S P,
# where S is the predicted tokens (one-hot vectors)
def _build_validity_mats(module_names):
    state_size = 3
    num_vocab_nmn = len(module_names)
    num_constraints = 4
    P = np.zeros((num_vocab_nmn, state_size), np.int32)
    W = np.zeros((state_size, num_vocab_nmn, num_constraints), np.int32)
    b = np.zeros((num_vocab_nmn, num_constraints), np.int32)

    # collect the input and output numbers of each module
    att_in_nums = np.zeros(num_vocab_nmn)
    att_out_nums = np.zeros(num_vocab_nmn)
    ans_out_nums = np.zeros(num_vocab_nmn)
    for n_s, s in enumerate(module_names):
        if s != '<eos>':
            att_in_nums[n_s] = _module_input_num[s]
            att_out_nums[n_s] = _module_output_type[s] == 'att'
            ans_out_nums[n_s] = _module_output_type[s] == 'ans'
    # construct the trasition matrix P
    for n_s, s in enumerate(module_names):
        P[n_s, 0] = att_out_nums[n_s] - att_in_nums[n_s]
        P[n_s, 1] = ans_out_nums[n_s]
        P[n_s, 2] = -1
    # construct the validity W and b
    att_absorb_nums = (att_in_nums - att_out_nums)
    max_att_absorb_nonans = np.max(att_absorb_nums * (ans_out_nums == 0))
    max_att_absorb_ans = np.max(att_absorb_nums * (ans_out_nums != 0))
    for n_s, s in enumerate(module_names):
        if s != '<eos>':
            # constraint: a non-<eos> module can be outputted iff all the following holds
            # * 0) there's enough att in the stack
            #      #att >= att_in_nums[n_s]
            W[0, n_s, 0] = 1
            b[n_s, 0] = att_in_nums[n_s]
            # * 1) for answer modules, there's no extra att in the stack
            #      #att <= att_in_nums[n_s]
            #      -#att >= -att_in_nums[n_s]
            #      for non-answer modules, T_remain >= 3
            #      (the last two has to be AnswerType and <eos>)
            if ans_out_nums[n_s] != 0:
                W[0, n_s, 1] = -1
                b[n_s, 1] = -att_in_nums[n_s]
            else:
                W[2, n_s, 1] = 1
                b[n_s, 1] = 3
            # * 2) there's no answer in the stack (otherwise <eos> only)
            #      #ans <= 0
            #      -#ans >= 0
            W[1, n_s, 2] = -1
            # * 3) there's enough time to consume the all attentions, output answer plus <eos>
            #      3.1) for non-answer modules, we already have T_remain>= 3 from constraint 2
            #           In maximum (T_remain-3) further steps
            #           (plus 3 steps for this, ans, <eos>) to consume atts
            #           (T_remain-3) * max_att_absorb_nonans + max_att_absorb_ans + att_absorb_nums[n_s] >= #att
            #           T_remain*MANA - #att >= 3*MANA - MAA - A[s]
            #           - #att + MANA * T_remain >= 3*MANA - MAA - A[s]
            #      3.2) for answer modules, if it can be decoded then constraint 0&1 ensures
            #           that there'll be no att left in stack after decoding this answer,
            #           hence no further constraints here
            if ans_out_nums[n_s] == 0:
                W[0, n_s, 3] = -1
                W[2, n_s, 3] = max_att_absorb_nonans
                b[n_s, 3] = 3*max_att_absorb_nonans - max_att_absorb_ans - att_absorb_nums[n_s]
        else:  # <eos>-case
            # constraint: a <eos> token can be outputted iff all the following holds
            # * 0) there's ans in the stack
            #      #ans >= 1
            W[1, n_s, 0] = 1
            b[n_s, 0] = 1

    return P, W, b

class Assembler:
    def __init__(self, module_vocab_file):
        # read the module list, and record the index of each module and <eos>
        with open(module_vocab_file) as f:
            self.module_names = [s.strip() for s in f.readlines()]
        # find the index of <eos>
        for n_s in range(len(self.module_names)):
            if self.module_names[n_s] == '<eos>':
                self.EOS_idx = n_s
                break
        # build a dictionary from module name to token index
        self.name2idx_dict = {name: n_s for n_s, name in enumerate(self.module_names)}
        self.num_vocab_nmn = len(self.module_names)

        self.P, self.W, self.b = _build_validity_mats(self.module_names)

    def module_list2tokens(self, module_list, T=None):
        layout_tokens = [self.name2idx_dict[name] for name in module_list]
        if T is not None:
            if len(module_list) >= T:
                raise ValueError('Not enough time steps to add <eos>')
            layout_tokens += [self.EOS_idx]*(T-len(module_list))
        return layout_tokens

    def _layout_tokens2str(self, layout_tokens):
        return ' '.join([self.module_names[idx] for idx in layout_tokens])

    def _invalid_expr(self, layout_tokens, error_str):
        return {'module': INVALID_EXPR,
                'expr_str': self._layout_tokens2str(layout_tokens),
                'error': error_str}

    def _assemble_layout_tokens(self, layout_tokens, batch_idx):
        # All modules takes a time_idx as the index from LSTM hidden states
        # (even if it doesn't need it, like _And), and different arity of
        # attention inputs. The output type can be either attention or answer
        #
        # The final assembled expression for each instance is as follows:
        # expr_type :=
        #    {'module': '_Find',        'output_type': 'att', 'time_idx': idx}
        #  | {'module': '_Transform',   'output_type': 'att', 'time_idx': idx,
        #     'inputs_0': <expr_type>}
        #  | {'module': '_And',         'output_type': 'att', 'time_idx': idx,
        #     'inputs_0': <expr_type>,  'inputs_1': <expr_type>)}
        #  | {'module': '_Answer',      'output_type': 'ans', 'time_idx': idx,
        #     'inputs_0': <expr_type>}
        #  | {'module': INVALID_EXPR, 'expr_str': '...', 'error': '...',
        #     'assembly_loss': <float32>} (for invalid expressions)
        #

        # A valid layout must contain <eos>. Assembly fails if it doesn't.
        if not np.any(layout_tokens == self.EOS_idx):
            return self._invalid_expr(layout_tokens, 'cannot find <eos>')

        # Decoding Reverse Polish Notation with a stack
        decoding_stack = []
        for t in range(len(layout_tokens)):
            # decode a module/operation
            module_idx = layout_tokens[t]
            if module_idx == self.EOS_idx:
                break
            module_name = self.module_names[module_idx]
            expr = {'module': module_name,
                    'output_type': _module_output_type[module_name],
                    'time_idx': t, 'batch_idx': batch_idx}

            input_num = _module_input_num[module_name]
            # Check if there are enough input in the stack
            if len(decoding_stack) < input_num:
                # Invalid expression. Not enough input.
                return self._invalid_expr(layout_tokens, 'not enough input for ' + module_name)

            # Get the input from stack
            for n_input in range(input_num-1, -1, -1):
                stack_top = decoding_stack.pop()
                if stack_top['output_type'] != 'att':
                    # Invalid expression. Input must be attention
                    return self._invalid_expr(layout_tokens, 'input incompatible for ' + module_name)
                expr['input_%d' % n_input] = stack_top

            decoding_stack.append(expr)

        # After decoding the reverse polish expression, there should be exactly
        # one expression in the stack
        if len(decoding_stack) != 1:
            return self._invalid_expr(layout_tokens, 'final stack size not equal to 1 (%d remains)' % len(decoding_stack))

        result = decoding_stack[0]
        # The result type should be answer, not attention
        if result['output_type'] != 'ans':
            return self._invalid_expr(layout_tokens, 'result type must be ans, not att')
        return result

    def assemble(self, layout_tokens_batch):
        # layout_tokens_batch is a numpy array with shape [T, N],
        # containing module tokens and <eos>, in Reverse Polish Notation.
        _, N = layout_tokens_batch.shape
        expr_list = [self._assemble_layout_tokens(layout_tokens_batch[:, n], n)
                     for n in range(N)]
        expr_validity = np.array([expr['module'] != INVALID_EXPR
                                  for expr in expr_list], np.bool)
        return expr_list, expr_validity
