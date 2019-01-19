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

We want something that given a sentence, it searches thru lots of sentences and it retrieves the word corresponding to that definition.

We'd like to start with almost a perfect accuracy on the task of remembering 10 sentences. The task is to retrieve them selectively. it'd be great to do this with as little data as possible, though Idk how I can achieve that.

Bare comparison of the sentences, gives around .5 in dev data.
Next steps: better comparison, larger dataset.

So: we want to compare two sentences.
sentence = [WE(word) for word in sentence]

def similarity(e1, e2):
    #.1
    out = e1.T.dot(e2)

    #.2
    out = MLP([e1, e2, e1 - e2, e1 * e2, e1Ae2])

    return out

#1
def encode(sentence):
    states = np.zeros((seq_length, embeddings_size))
    for i in range(num_rnns):
        states[i], _ = BiRNN_i(sentence)
    #states is num_rnns x seq_length x 2 x hidden_size
    transform(states)
    #states is num_rnns x seq_length x hidden_size

    #1.1 (we could also calculate probs with another BiRNN)
    probs, states = states
    #states is num_rnns x seq_length x (hidden_size - 1)
    #probs is num_rnns x seq_length x 1
    #1.1.1
    state = softargmax(states, p=probs)
    #1.1.2
    state = max(states, p=probs)

    #1.2
    _, right_state = RNN_left(sentence)
    _, left_state = RNN_right(sentence)
    state = concat(left_state, right_state)
    state = MLP(state)

    return state

#2
def encode(sentence):
    starting_points = [0, 3, ..., len(sentence) - 1] #this 'barredores' could be evenly spaced.
    out = np.zeros()
    for i in range(n):
        _, final_state = RNN_i(sentence, starting_point=starting_points[i])
        out[i] = final_state
    return out

similarity(encode(s1), encode(s2))

#experiment: how useful is to have (a - b), aAb, a * b? Compare usefulness for different tasks (this could be interesting to do in Ashwin thing)
max pool, softargmax, is maxpool differentiable?

__ Next tasks:
* generate a larger dataset
* get a simple way of comparing sentences which is better than comparing word by word.
    * first: pass an rnn through all the data and compute the similarity between the two hidden states.
    * first point twentyfive: pass a rnn through all the data and instead of just keeping the last state of the rnn, compute a softmax in all the states.
    * first point twentyfive point 0625: use the left and right final states of the rnns to compute the similarity, not just the right state
    * first point five: pass a birnn through all the data and compute the bla bla thing
    * second: the #1 above

make a model that counts the similar words between each def and def'. then returns def wiuth max count. also try avoidinig counting common words

Note about the training set: there are some instances of the task that are impossible. We can take this as an advantage in the following way. We can let the model say "I don't know" and we just skip that example. we don't know the specific amount of examples that the model can't know, but we can test a human on 300 examples and calculate the proportion of IDKs. That quantity will probably extrapolate to the general case. The interesting thing though is that the ones that do make sense are similar but do not share that many words. And even if they share some words, it isn't easy to tell that they are referring to the same word. Eg
> the branch of engineering that deals with things smaller than 100 nanometers (especially with the manipulation of individual molecules
the science of developing and making extremely small but powerful machines

Here we could say that the only non-common word shared is smaller and small, but nothing else.

And there would probably be some other sentence that starts with the same words as the second sentence above. (eg "the science of developing powerful machines that are useful for large computations.")

Try combining the birnn representation before concatenating everything to the nn. Also, think whether this could be better/worse/equal to just adding a new layer. Pros: left and right representations could be good to be combined, they together have information of all the sentence. Cons: we are losing useful information eg we can't directly compare left rnn representaiton in position i with that of position i+1. That could give us useful information about what happened in a specific word.

is it possible/useful to go from char-level representation to word embeddings? how would that look like

everythign is great after meeting people :)
** espero no pasarme toda la vida poniendome buzos **

what does it mean that tf.layers.dense((2x3x4), 5) yields 2x3x5. start thinking about it in 2d and then go to 3d and then 17d (or xd for that matter :)

It would be interesting to work in some model of memory that is stable - - that is easy to train - - not like the ntm

We should allow multiple passes, for instance if the def is "past participle of smell" then the model needs to look up for smell.


As we recollect our data, we can test how the model performs with little data. We have around 1.5k cases. Using memsize of 4 gives us 375 cases.

Let's debug our implementation of the 1 + 4 methods by seeing if they can overfit to 100 cases.

With around 1.8K words scraped (around 900) the model overfits to the data.  

Params: Model 0: 6k: mem_size: 8
Results: train = 1, dev_max = .45, dev_stable = .41

My guess is that model 0 is easier and faster to train. However, it's very limited.


We can think of books as discrete points in a very high-dimensional space. Eg there would be clusters of books that have similar topics. Consider the space as continouos. It would be great to have the "general book" where you can input some specific value for each axis, and a book is produced (I imagine it still being with discrete words. What's continouos is the _meaning_ of the book.)

# 18k test
## -1 (this is the same result for one batch for dev as for ten batches)
stable after 20k iterations
train: 1. dev: .675
trained for around 75k
### Comments
This works surprisingly well, taking into account that we are just comparing words that are in the same position. Let's understand how it works by seeing where it's correct/where it fails.


## 0
stable after 15 iterations
train: 1. dev: .54
(This is surprising)
trained for around 75k

## 1.1


Before going to more complex models, it makes sense to understand the performance of the simpler models. Without doing this, we can't know whether the more complex model is good or not.

Problem 1: we aren't normalizing by the length of the sentence. We aren't comparing

My theory on why the models 0 and 1 don't generalize well to dev set. They are much more complex, so they can easily overfit to the training data without that much learning. (and they do overfit, as we see )

steps
- get a larger dataset (download two dicts)
- test different models -1, 0, 1.1, 1.2
- what if we add word embeddings?
(Make the experiments such that I could tell what's the result and I can leave them runnign for a long time.)
- Little data
- Understand how the model -1 reaches that performance.

Memories (and returing the correct output,) seems to be a chain reaction. Given the incorrect first memory, and you can spend minutes trying to come up with the memory, even if it was something as easy as a password that I've written around 1k. It is related to being a linked list.  


Also for the ideas: the net of interconnected concepts we were thinking with nico

Next steps:

The opted-to-json I have in /tests

https://github.com/Barjak/OED_unpack_tools.py

https://github.com/shenfeng/dictionary/tree/master/src


import json
import string
decoder = json.JSONDecoder()

defs = {}
for char in string.ascii_lowercase:
    defs.update(decoder(f'OPTED-to-JSON/json/{char}.json'))

var fs = require('fs');
fs.readFile('Adjective.js', 'utf8', function(err, contents) {
    content = JSON.stringify(contents)
});
fs.writeFile("adjective.json", content, function(err) {
    if(err) {
        return console.log(err);
    }

    console.log("The file was saved!");
});


Future: wiktionary + wikidata. (this should be a large dataset, but only start with this after playing more with the architectures.x it doesn't make sense to spend so much time in the dataset.)

Now it's time to spend on the code and not on datasets. For the future it could be nice to do (webster | wiktionary) + oxford scraped

Compare the perforamnce of datasets of 54k vs 18k. In that way we can see how a dataset of 200k will behave. Also, enabling the model to lookup for more definitions will be great.


So we know it's overfitting. In particular, the dev performance dropped from around 0.36 to 0.32.

The MLP has around 1m parameters. The interesting thing to consider is the number of parameters of the word embedding. oops. It has .1M * 500 = 50M. Now the overfitting seems more than reasonable, given that we have 50k words, and 6k cases. It also seems interesting that the dev acc isn't that bad. It would be interesting to know what cases is the nn getting right. The overfitting doesn't seem to be caused just by the word embeddings size: I try reducing it to 16 and 80 and the dev acc got lower, not higher, while the training accuracy remained in a similar value (around .97)

Exp1: what cases is the nn getting right?
exp2: word embeddings to 16 instead of 512. something particular of this dataset is that it could have several words that appear only so many times. so it can do whatever it wants with them. Also: try without training the WE at all. Also,

when embeddings arrive
* try -1 mode without training
* try to make 0 mode as good as possible.


how good is jag10 from jag20? drew?
sign to slack nlp account

Lesson: it doesn't seem that good to work on my own datasets if there are other datasets that can fulfill my requirements. Creating a dataset takes a lot of time, and we also can't compare with other's work.

## Experiment: comparing sentences word by word
In this experiment, we use one of the most simple methods of comparing sentences. We first compute a representation for each word and then we compute the similarity between the corresponding word of each sentence. For instance, say we are comparing "This is a bottle" with "There is a can." The first thing we compare is the representation for the word "there" with that of "this." Then we compare "is" with "is." And so on.

This is clearly an incorrect approach. For instance, "Clearly, it is a cat" and "It is a cat" would be very different in this similarity measure, but they shouldn't. However, it could work as a baseline.

The only decision we can make here is how we initialize the representations for the words (ie the word embeddings.) If we don't train the embeddings, a random initialization gives almost the same performance as initializing it with the GloVe word embeddings. The performance is slightly above chance (performance: 13.0% - 13.5%. chance: 12.5%). It seems weird to me that GloVe doesn't add any improvement over random vectors. This could be because GloVe isn't correctly working.

## Thought
It could make sense to continue with a simpler task.

Lesson: from now on, try to be more organized in this notes and the code I write. Not that much, though. 