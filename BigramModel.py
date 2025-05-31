import csv

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from contourpy.array import outer_offsets_from_list_of_offsets

torch.set_printoptions(sci_mode=False)

# read file to words
words = open('../Popular_Baby_Names(only names).txt', 'r').read().lower().splitlines()
names = open('../us.txt', 'r').read().lower().splitlines()

# inits array to store count values
Char_Count_Array = torch.zeros((27,27), dtype=torch.int32)

# creates lookup table for converting string into ints
Char_Lookup = sorted(list(set(''.join(words))))

# converts character strings to ints
Char_To_Int = {s:i for i, s in enumerate(Char_Lookup)}
Char_To_Int['.'] = 0
Int_To_Char = {i:s for s, i in Char_To_Int.items()}


# Array of probabilities for a given bigram
Probability = (Char_Count_Array+1).float() # '+1' for model smoothing
Probability = Probability / Probability.sum(1, keepdims=True)


# creates training set of all bigrams (x = first character,y = next character to be determined)
xs, ys = [], []
for word in words:

    chars = ['.'] + list(word) + ['.']

    for letter1, letter2 in zip(chars, chars[1:]):

        intx1 = Char_To_Int[letter1]
        intx2 = Char_To_Int[letter2]
        xs.append(intx1)
        ys.append(intx2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

g = torch.Generator().manual_seed(2147483647)
# creates random weights for 27 x N(numNeurons)
weights = torch.randn(27, 27, generator=g)
weights.requires_grad_(True)



# encodes xs to a vector that can be sent into the neural network
# One Hot Encoding
xEncoded = F.one_hot(xs, num_classes=27).float()

# matrix multiplication for determining weights given an input
# [len(xs), 27] X [27, 27] ---> [len(xs), 27]
for i in range(100):

    # --------------Forward Pass--------------#
    # --------------Forward Pass--------------#
    # --------------Forward Pass--------------#
    logits = xEncoded @ weights # log-counts
    counts = logits.exp() # equivalent to Char_Count_Array
    prob  = counts / counts.sum(1, keepdims=True) # probability distribution
    loss = -prob[torch.arange(len(xs)), ys].log().mean()
    # counts and prob are also called a softmax

    #--------------Backward Pass--------------#
    #--------------Backward Pass--------------#
    #--------------Backward Pass--------------#
    weights.grad = None
    loss.backward()

    # update weights
    weights.data += -50 * weights.grad

#    print(loss.item())

g = torch.Generator().manual_seed(21474837)

for i in range(5):
    outList = []
    index = 0
    while True:
        xEncoded = F.one_hot(torch.tensor([index]), num_classes=27).float()
        logits = xEncoded @ weights
        counts = logits.exp()

        probability = counts / counts.sum(1, keepdims=True)
        index = torch.multinomial(probability, num_samples=1, replacement=True, generator=g).item()
        outList.append(Int_To_Char[index])
        if (index == 0):
            break
    print(''.join(outList))



# plots bigram frequency
def Plot_Bigram_Frequencies():
    plt.figure(figsize=(16,16))
    plt.imshow(Char_Count_Array, cmap='Blues')
    for i in range(27):
        for j in range(27):
            chstr = Int_To_Char[i] + Int_To_Char[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
            plt.text(j, i, Char_Count_Array[i, j].item(), ha="center", va="top", color='gray')
    plt.axis('off')
    plt.show()

