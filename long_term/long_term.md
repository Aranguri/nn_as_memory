# Next tasks
1) Clean life (aka code)
2) Instead of giving the system n new memories each time, give it n fixed memories and then ask it to retrieve them.


1) Given <w1, def1>, <w2, def2>, and <?, defn'> predict wn. We compare def1 and def2 with defn'
2) Instead of writing in memory, show it to the mlp. Then, it decides what to do with the memory (we can enlarge the distance between presented memory and request of that memory as times goes on. And we thus can explore how|good BPTT works.) I think we can use the NTM mechanism for this.

Compare efficiency mse and selecting a vector with cesoftmax and having a prob distr.

## Unrelated tasks
* are glove unit vectors

# Idea
Give a neural net several <word, definition> pairs. Then, we give it a query <?, definition'>  with definition' being another definition to refer to a word.
If it doesn't work, we can use definition = definition' (and limit the memory to be small)
Make somethign easy that works and build from there.
Output: vector in 50 dims. (Other possibility: char-by-char generator)

Baseline task: don't use <word, definition> at all. Just train an LSTM with word embeddings on the query <?, definition'> and measure the distance between the
 result of the query and the ground truth (to understand this distance, we can measure the distances between two random vectors, and the distances between the
 words in the definition and the ground truth word.) Also, we can measure the accuracy.

* we can try with making it possible to backprop through the word embeddings.
* say we have one dataset of 100M images. Is it always the case that we want to train on new data? Or seeing the same data again could help?
* try semantic average
* add this to a file
sudo -i
# sync; echo 1 > /proc/sys/vm/drop_caches
# sync; echo 2 > /proc/sys/vm/drop_caches
# sync; echo 3 > /proc/sys/vm/drop_caches
* make a single file for utils
* is it possible to import a file with imports

Bidirectional RNNs:
* we create a forward RNN and a backward RNN (which receives the input reversed.) The output of the BiRNN is just the concatenation of the output of both RNNs.
* other option: compute forward and backward pass as before, but instead of returning the result, use that as the states for the previous states. That is, in the left-most entry, we use as previous state the last state computed by the backwrad rnn. For the right-most entry, we use the last state from the forward rnn. For one state in the middle, we use one prev hidden state from the forward and one from the backward.

## Similarity
How do we measure similarity between sentences? The similarity measure shouldn't be local. The two sentences
"the cat is under the table, and the dog is running"
"the dog is running, and the cat is under the table"
are semantically almost equal, but they almost don't share words in the same positions.

Say we use a rnn to compress the sentence into one state. Probably, we are losing a lot of information. A model can't write everything in one vector.
One way to do this is by using n rnns. Thus, as we get an output for each rnn, we concatenate n of them and we have a distributed representation of the input.
What I don't like about RNNs is the focus on the last (or first) words.

We can also work with the example usage of the word. Ie, give the usage we need to guess the word given that we had the definition (eg we have to build a function that goes from usage (with a blank in the word) to definition to look up the word.)

Also: try with _very large_ memories. Like 1M. What new structures do we need to add?

## Memory ops - How does the mlp decide what to store?
### RNNs
h_t = tanh(Wh_{t-1} + Vx_t)
y_t = Uh_t
The memory is the hidden state.
Adding a memory: entries in V that have non-zero values
Removing a memory: entries in W that have zero values
Retrieving a memory: entries in U that have non-zero values

### External Memory


## Logs
* Min dev dist: .39 (lr 1e-2, h_size 50, bilstm)
* Min dev dist: .35 (lr 1e-4, h_size 200, bilstm @2000)
* Min dev dist: .27 (lr 1e-4, h_size 200, bilstm @6000)
* Min dev dist: .25 (lr 1e-4, h_size 200, bilstm +@10000)
* Min dev dist: .32 q   5 (lr 1e-4, h1:100, h2:100, ffnn, @2000)
* Min dev dist: .28 (lr 1e-4, h1:100, h2:100, ffnn, @12000)
* Min dev dist: .27 (lr 1e-4, h1:100, h2:100, ffnn, +@20000)
