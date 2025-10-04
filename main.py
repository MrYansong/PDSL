from PDSL import run
import numpy as np
import time
import os, sys
import argparse
os.chdir(sys.path[0])

class Logger(object):
    def __init__ (self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write (self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush (self):
        # self.log.flush()
        pass

parser = argparse.ArgumentParser(description="PDSL")
datasets = ['jazz_SI', 'jazz_GLT', 'jazz_SIR', 'netscience_SI', 'netscience_GLT', 'netscience_SIR',
            'CollegeMsg_SI', 'CollegeMsg_GLT', 'CollegeMsg_SIR', 'cora_SI', 'cora_GLT', 'cora_SIR',
            'BA_SI', 'BA_GLT', 'BA_SIR']
parser.add_argument("--dataset", default='jazz_SI', type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
parser.add_argument( "--train_ratio", default=0.9, type=float, help="The ratio of train sets")
parser.add_argument( "--encoder_hidden", default=512, type=int, help="")
parser.add_argument( "--encoder_latent", default=256, type=int)
parser.add_argument( "--ODE_outdim", default=5, type=int)
parser.add_argument( "--ODE_t_size", default=2, type=int)
parser.add_argument( "--MLP_hidden", default=[128, 128, 128], type=list)
parser.add_argument( "--max_train_epochs", default=100, type=int)
parser.add_argument( "--inference_epochs", default=30, type=int)
parser.add_argument( "--learning_rate", default=2e-3, type=float)
parser.add_argument( "--inference_learning_rate", default=5e-2, type=float)


args = parser.parse_args(args=[])
scores = []
sys.stdout = Logger(args.dataset + '.txt')
for i in range(10):
    start = time.time()
    precision, recall, f1, acc, auc_float, auc_binary = run(args)
    scores.append([precision, recall, f1, acc, auc_float, auc_binary])
    end = time.time()
    remaining_hours = (end - start) / 3600 * (5 - i - 1)
    print('++++++++++++++++++++++++++++++ This snapshot remaining hours: ', "{:.2f}".format(remaining_hours), ' hours +++++++++++++++++++++++++++')

print('scores: precision, recall, f1, acc:')
for scor in scores:
    print("Test finished","\tPrec: {:.5f}".format(scor[0]),
                "\tRec: {:.5f}".format(scor[1]),
                "\tF1: {:.5f}".format(scor[2]),
                "\tAcc:{:.5f}".format(scor[3]))

scores_array = np.array(scores)
ave_scores = np.mean(scores_array, axis=0)
# print("snaps_number: ", str(j))

print(" Average test scores:","\tPrec: {:.5f}".format(ave_scores[0]),
        "\tRec: {:.5f}".format(ave_scores[1]),
        "\tF1: {:.5f}".format(ave_scores[2]),
        "\tAcc:{:.5f}".format(ave_scores[3]))