uv is used for installation

Install all dependencies with `uv sync --editable`

### Some thougts:

- Prior over spikes p(s) critical, one could use data driven priors or the very simple priors encoded in the generative models (e.g. two state hidden markov), but the cool thing when doing SBI: This prior can be anything/completely intractable as long as one has samples.
- To simplify, its probably a good idea to model the model conditional expectations:
That is, instead of modeling the full p(theta, s | x), one could  model p(theta | x), and then E(s | x, theta).
The latter would just be a deterministic network (as is used in Cascade), that gets some global parameters and the indicator trace.
- As far as I can see, in Cascade, the ground truth spikes are smoothed. That is the target is not the  discrete spike train, but a Gaussian smoothed version of it.
- While it seems to kinda work in early experiments, working directly on these very long time series is probably making life way to difficult given that most of whats going on is quite local in time. Cascade works with a very short window.
One could think of doing generatively modeling autoregressively, but with a finite history.


### Questions

- Why does cascade use such a small window? 64 or 128? Are they using a different sampling rate?
- What models do there exist? We have the simple Hill, the one by Greenberg, the one by Broussard, which is a more advanced version of the one by Giovanni. There are other model based methods. What models are they using?
