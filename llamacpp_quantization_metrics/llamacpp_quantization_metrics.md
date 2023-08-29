# llama.cpp Quantization Metrics

Language models are usually trained as 16 bit floats but for inference they are frequently *quantized* to lower precision which results in both faster inference and reduced memory use.
A typical quantization format uses something like 2-8 bits per weight (BPW) for storing the information of the unquantized model.
However, this quantization and subsequent loss in precision lowers the quality of the model performance.
This can be observed subjectively but especially in order to optimize quantization schemes an objective measure of quality loss is desirable.
Currently a popular approach is to calculate the *perplexity* of both the quantized and unquantized models on a large corpus of text and to optimize the increase of the perplexity of the quantized model.
However, this blog post aims argue that this is a suboptimal metric and that the *root mean square* of the *token probabilities* is a better metric.

Let us start with considering how a model produces a probability distribution for all possible $N$ tokens in its vocabulary.
First the model puts out raw *logits* $l_n$ for each token.
These logits cannot be interpreted sensibly on their own but they can be converted to probabilities by using *softmax*:

$$p_n := \mathrm{softmax}(l_n) = \frac{\exp(l_n) \quad}{\sum_{j=1}^N \exp(l_j)} .$$

From these probabilities the model can then sample to produce a prediction for the next token.
To calculate perplexity the model is instead given a text consisting of $M$ tokens and the probability that it assigned to the "correct" token is compared to its predictions:

$$\mnathrm{PPL} = \exp \left( \frac{- \ln p_{nm}}{M} \right)$$.

So effectively the perplexity is the exponential of the average *negative log-likelihood* per token.
The perplexity is by definition positive and a higher value corresponds to better predictions.
Without further investigation we can also notice an asymmetry:
for $p_{nm} \rightarrow 1$ the perplexity is bounded at 0 but for $p_{nm} \rightarrow 0$ it can grow indefinitely.

If we now quantize one or more tensors of the model this necessarily introduces *rounding error* somewhere in our model.
If the rounding error is small enough then it will simply propagate to the logits and add a bit of noise there.
So let us begin by looking at the difference in the correct logits for Q8_0 which uses 8.5 BPW relative to F16:

![Q8_0 logit diffs](plots/logit_diff_hist/logit_diff_hist_q8_0.png)

The logit differences mostly follow a Gaussian distribution (the tails are too fat) and the mean of the distribution is close to 0.
The distribution also only weakly depends on the output probability of the correct token for the unquantized model:

![Q8_0 logit mean std](plots/logit_mean_std/logit_mean_std_q8_0.png)

For this plot the data was histogrammed to 10 equal-sized bins of size 0.1.
The markers indicate the means of the values for a given bin while the error bars indicate their standard deviation.
The difference in logits seems to be larger for those tokens where the F16 probability was highest but the mean difference is still much smaller than the standard deviation of the differences.
On the other extreme end, this is what the corresponding plots look like for Q2_K (3.35 BPW on average across the model):

![Q2_K logit diffs](plots/logit_diff_hist/logit_diff_hist_q2_k.png)

![Q2_K logit mean std](plots/logit_mean_std/logit_mean_std_q2_k.png)

For this quantization format the first thing to notice is that the absolute differences in logits are much larger which should not be surprising given the much larger rounding error.
But what is also noticeable is that the mean of the distribution has significantly shifted downwards, especially for tokens where the unquantized model performed well.
This general trend is mostly continuous as the BPW of the quantization scheme are varied.
However, there are some weird edge cases like Q5_0 where the correct logits on average actually *increase*:

![Q5_0 logit mean std](plots/logit_mean_std/logit_mean_std_q5_0.png)

Regardless, both subjectively and objectively (as measured by perplexity) Q5_0 performs worse than F16.
This is because any kind of noise on the logits is detrimental.
The more noise there is the more does the distribution of token probabilities get pushed towards a *uniform distribution* where each token has a constant and equal probability.
But at this point we should remind ourselves that the actual values of logits are not relevant - what matters are the probabilities that the model assigns to each token.
So let us look at how quantization affects the correct token probabilities for Q8\_0 and Q2\_K:


![Q8_0 prob diff hist](plots/prob_diff_hist/prob_diff_hist_q8_0.png)

![Q2_K prob diff hist](plots/prob_diff_hist/prob_diff_hist_q2_k.png)

We can notice that the peak for q8\_0 is much sharper than for Q2\_K and that the Q2_K distribution is skewed towards negative values (the quantized model performing worse). However, for both quantization types the most likely change in token probability is little to no change at all.
The reason for this becomes apparent if we look at the token probability distribution of the unquantized model:

![F16 prob hist](plots/prob_hist_f16.png)

The token probabilities have two clusters: one close to 0 and one close to 1.
In those regions softmax is very flat so changes in the logits result in only small changes in the token probabilities.
Now, let us think back to the beginning of this blog: the goal is to find a good metric for measuring the quality loss from quantization.
The metric proposed here is the *root mean square** of the differences in token probabilities:

$$\mathrm{RMS}_p^2 = \frac{ \sum_{m=1}^M (p_{nm}^\mathrm{Q} - p_{nm}^\mathrm{F16})^2 }{ M },$$

where the upper indices indicate whether the probabilities correspond to the unquantized F16 or the quantized model.
Effectively this metric calculates the standard deviation of the differences assuming that the mean is of the differences is 0.
This approximation is unproblematic because the mean of the differences is small relative to their standard deviations (lowest ratio is for Q2_K where the standard deviation is ~4 times larger than the absolute value of the mean).
If we now compare plots of the perplexities and the RMS of the different quantization formats we find:

![Combined plot PPL](plots/combined/combined_perplexity.png)

![Combined plot PPL](plots/combined/combined_rms_probs.png)

Though there are some differences (see the Q5 values) the plots overall look rather similar.
So why would you prefer one metric over the other.
The first reason is that the two metrics are not equally sensitive to the same areas of token probability.
Perplexity sharply increases when token probabilities get close to 0.
Pair this with the fact that there are a lot of low-probability tokens in the first place and you find that the perplexity value is overwhelmingly dominated by low-probability tokens:

![F16 prob hist](plots/ppl_contributions_f16.png)

This is a problem if you consider top-p sampling.
With this sampling method low-probability tokens would never be considered for sampling in the first place.
So changes on the probabilities of those tokens would not make any difference for the actual generation of text.
If we look at the differences in perplexity when calculated on only a small subset of the total tokens (e.g. the tokens with a probability between 99% and 100%) we find that the differences in perplexity are also the largest for low-probability tokens:

![PPL diff Q8_0](plots/ppl_diff/ppl_diff_q8_0.png)

However, there are some weird things happening for some of the quantization formats:

![PPL diff Q2_K](plots/ppl_diff/ppl_diff_q2_k.png)

There are instances where the quantized models have a better perplexity on the lowest-probability bin < 1%.
I'm not entirely sure what is happening here but I hypothesize that the unquantized model is confidently wrong about something in the text.
Through the process of quantization that confidence is then sometimes eroded away and the model is punished less severely for its wrong confidence - or there is simply a bug in my code.
In any case, you can completely suppress this effect by filtering out all tokens where the unquantized model has a probability < 0.1% of being correct (these account for 4.2% of all tokens):

![PPL diff Q2_K corrected](plots/ppl_diff/ppl_diff_corrected_q2_k.png)

Filtering out these very low-probability tokens also reduces how much higher the perplexity difference for Q8_0 is for the lowest-probability bin:

![PPL diff Q8_0 corrected](plots/ppl_diff/ppl_diff_corrected_q8_0.png)

By comparison, the contributions to $\mathrm{RMS}_p$ mostly come from medium-probability tokens:

![RMS contribution Q8_0](plots/prob_rms_contribution/prob_rms_contribution_q8_0.png)

Notably all contributions to $\mathrm{RMS}_p$ are by definition non-negative.
Unlike with a difference in perplexity, it is therefore impossible for multiple contributions to cancel each other out.
For quantization types that use less than 4 BPW for all layers (Q2\_K, Q3\_K\_S) there are large $\mathrm{RMS}_p$ contributions from high-probability tokens:

![RMS contribution Q2_K](plots/prob_rms_contribution/prob_rms_contribution_q2_k.png)

Still, this is less lopsided than the equivalent situation with perplexity.
My personal interpretation of the plots is that Q2\_K and Q3\_K\_S begin to fail a lot of tokens that the unquantized model was able to get correct with high confidence.
For comparison, this is the plot for Q3\_K\_M:

![RMS contribution Q3_K_M](plots/prob_rms_contribution/prob_rms_contribution_q3_k_m.png)

There is still a comparatively high contribution from high-probability tokens but it is not disproportionally higher.
Now, so far we have only talked about the metrics in and of themselves but there are also practical differences.
When calculating perplexity we are by definition limited to the "correct" tokens that actually appear in the text that we use as input.
If we are only concerned with replicating the output of the unquantized model as closely as possible though we could instead look at the probability differences of any token.
It should also be possible to generalize $\mathrm{RMS}_p$ for more than one token probability per token from the input text.
You could for instance, compare the probabilities of the 10 most likely tokens for the unquantized model.
This would give you ten times more data points per token of input text and significantly reduce the amount of input text required to get sufficient precision.
For reference, the uncertainty of $\mathrm{RMS}_p$ can be estimated to be ~1% if you assume a Gaussian distribution.

The way to implement calculating $\mathrm{RMS}_p$ from more than one input token could be as follows:
First, calculate reference probabilities using the unquantized model.
During the calculation, select some subset of probabilities and write them to a file (combinations of tokens and their assigned probabilities).
Next, re-run the calculations with the quantized model and provide the file as input.
You can now select the tokens specified in the file and calculate $\mathrm{RMS}_p$ for each token.
The final result would then just be the sum of $\mathrm{RMS}_p$ for all tokens.

One issue that could arise from calculating more than one $\mathrm{RMS}_p$ value per input token is that due to the nature of softmax the probabilities will be correlated.
So the rate of convergence towards the final value that you would see with infinite tokens should be slower than just the rate of convergence with 1 token multiplied with the number of tokens considered.
The rate of convergence should always increase by adding more tokens however (but you will need to be careful when trying to assign an uncertainty to the final result).

Finally, what I did not explore is the portability of $\mathrm{RMS}_p$ across multiple input texts and models.
The perplexity calculated on two different input texts is going to be different, but $\mathrm{RMS}_p$ may end up being the same or at least very similar - this may be worth exploring.