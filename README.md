# matrixwalk

I created a little random walk agent that moves over a 2D or 3D character lattice and tries to predict the next character for a selected action (e.g. N,E,S,W for 2D) with a recurrent neural network. 

The agent can reach each grid position with a hugh number of different trace-histories .. nevertheless it would be sufficient to have one internal hidden-state for each unique grid position.

Observation:
I plotted the hidden state with t-SNE and found that no magic 'state compression' happens in a RNN - basically the RNN generates a lot of different hidden states for the same grid position depending on the history (it keeps as much context as is necessary to uniquely identify a position based on the last n observations, e.g. keeps a 'backlog' of 2 or 3 characters in most cases).

It might be desirable for a neural system to identify redundant states and compress them to a single internal representation: e.g. multiple histories all leading to the same grid positions could be treated equally so that the next character prediction would not have to be learned for each context individually but experience could be shared.

My impl is based on Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) but beside of the RNN models it was written from scratch to learn something about RNNs in [Torch](https://github.com/torch/torch7).

Next steps for this repo: Experiment with ideas for redundant state discovery and 'state compression'.
