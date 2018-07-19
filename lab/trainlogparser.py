import sys
import re
import matplotlib.pyplot as plt

trainAccuracy = []
crossEntropy = []
validationAccuracy = []

'''
INFO:tensorflow:2018-07-19 12:54:25.567241: Step 0: Train accuracy = 60.0%
INFO:tensorflow:2018-07-19 12:54:25.570262: Step 0: Cross entropy = 1.539507
INFO:tensorflow:2018-07-19 12:54:26.325964: Step 0: Validation accuracy = 41.0% (N=100)
'''


def main():
    with open("retrain.log", 'r') as logfile:
        for line in logfile:
            if ("Train accuracy" in line):
                line=line[69:]
                line.rstrip("%\n")
                line = re.sub(r"[\n\t\s%=]*","",line)
                if line == "100.0":
                    line == "99.99"
                trainAccuracy.append(line)
            if ("Cross entropy" in line):
                line=line[68:]
                line = re.sub(r"[\n\t\s%=]*","",line)
                crossEntropy.append(line)
            if("Validation accuracy" in line):
                line=line[74:]
                line = re.sub(r"[\n\t\s%=]*","",line)
                line=line[:-6]
                validationAccuracy.append(line)


    #print (trainAccuracy)
    #plt.plot(trainAccuracy)

    plt.plot(validationAccuracy)
    plt.ylabel("Validatation Accuracy % (N=100)")
    plt.xlabel("Accmulated Training Steps")
    plt.show()
 
if __name__== "__main__":
  main()

