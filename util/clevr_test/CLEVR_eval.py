from __future__ import print_function
import argparse
import json
from collections import defaultdict
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--questions_file', required=True)
parser.add_argument('--answers_file', required=True)


def main(args):
  # Load true answers from questions file
  true_answers = []
  with open(args.questions_file, 'r') as f:
    questions = json.load(f)['questions']
    for q in questions:
      true_answers.append(q['answer'])

  correct_by_q_type = defaultdict(list)

  # Load predicted answers
  predicted_answers = []
  with open(args.answers_file, 'r') as f:
    for line in f:
      predicted_answers.append(line.strip())

  num_true, num_pred = len(true_answers), len(predicted_answers)
  assert num_true == num_pred, 'Expected %d answers but got %d' % (
      num_true, num_pred)

  for i, (true_answer, predicted_answer) in enumerate(zip(true_answers, predicted_answers)):
    correct = 1 if true_answer == predicted_answer else 0
    correct_by_q_type['Overall'].append(correct)
    q_type = questions[i]['program'][-1]['function']
    correct_by_q_type[q_type].append(correct)

  for q_type, vals in sorted(correct_by_q_type.items()):
      vals = np.asarray(vals)
      print(q_type, '%d / %d = %.2f' % (vals.sum(), vals.shape[0], 100.0 * vals.mean()))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
