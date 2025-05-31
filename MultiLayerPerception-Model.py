import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import pickle as pkl
import time


torch.set_printoptions(sci_mode=False)

# read file to words
words = open('../Popular_Baby_Names(only names).txt', 'r').read().lower().splitlines()

# weights and biases from previous training sessions
oldWeights = pkl.load(open('weights.pkl', 'rb'))

# creates lookup table for converting string into ints
Char_Lookup = sorted(list(set(''.join(words))))

# converts character strings to ints
Char_To_Int = {s:i for i, s in enumerate(Char_Lookup)}
Char_To_Int['.'] = 0
Int_To_Char = {i:s for s, i in Char_To_Int.items()}

# building the dataset
blockSize = 3 # num characters to predict the next
def build_dataset(words):

    x, y = [], [] # x --> inputs to nn, y --> labels for nn

    for word in words:
        context = [0] * blockSize # inits array size based on num characters being used

        for char in word + '.':
            index = Char_To_Int[char]
            x.append(context)
            y.append(index)
            context = context[1:] + [index]

    x = torch.tensor(x)
    y = torch.tensor(y)

    return x, y



def setTrainedParameters():
    C = oldWeights[0]  # creates a 3 dimensional tensor
    W1 = oldWeights[1]  # numInputs, numNeurons in hidden layer
    b1 = oldWeights[2]  # biases
    W2 = oldWeights[3]  # inputs into final layer
    b2 = oldWeights[4]
    return C, W1, b1, W2, b2, [C, W1, b1, W2, b2]

def setRandomWeights():
    C = torch.randn((27, 10))  # creates a 3 dimensional tensor
    W1 = torch.randn((30, 200))  # numInputs, numNeurons in hidden layer
    b1 = torch.randn((200,))  # biases
    W2 = torch.randn((200, 27))  # inputs into final layer
    b2 = torch.randn((27,))
    return [C, W1, b1, W2, b2]

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

xTrainingSet, yTrainingSet = build_dataset(words[:n1]) # 80% of words
xDevSet, yDevSet = build_dataset(words[n1:n2]) # words from 80-90% of len(words)
xTestSet, yTestSet = build_dataset(words[n2:]) # last 10% of words

C, W1, b1, W2, b2, parameters = setTrainedParameters() # sets parameters

print(W1)

for p in parameters:
    p.requires_grad = True

# for testing learning rates
lre = torch.linspace(-3, -0.5, 1000)
lrs = 10**lre
lri = []
lossi = []
stepsi = []
learningRate = -0.1
startTime = time.time()
for i in range(200000):

    # mini batch construct
    index = torch.randint(0, xTrainingSet.shape[0], (100,))

    # forward pass
    embedding = C[xTrainingSet[index]]
    h = torch.tanh(embedding.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, yTrainingSet[index])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update parameters
    for p in parameters:
        p.data += learningRate * p.grad

    # track stats
    lossi.append(loss.item())
    stepsi.append(i)

    # learning rate decay
    if i < 100000:
        learningRate = -0.01
endTime = time.time() - startTime
print(loss.item())
print(endTime)

#plt.plot(stepsi, lossi)
#plt.show()

pkl.dump(parameters, open('weights.pkl', 'wb'))

# training split, dev/validation split, test split
# 80/10/10

for i in range(20):
    outList = []
    context = [0] * blockSize
    while True:
        embedding = C[torch.tensor([context])]
        h = torch.tanh(embedding.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim = 1)
        index = torch.multinomial(probs, 1).item()
        context = context[1:] + [index]
        outList.append(index)
        if index == 0:
            break
    print(''.join(Int_To_Char[char] for char in outList))
