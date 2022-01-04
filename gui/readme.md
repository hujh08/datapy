GUI for data

# Image Marker

Navigate and mark images one by one

## Brief of underlying logic

This marker consists of 3 parts -- graphical interface, images iterator and record system. **Graphical interface** is front end for the user. It displays image (and other informatin, e.g. image name, num of marked images), and respond to event triggered by the user, like key press. **Images iterator** fetches an image and passes to graphical interface. In the iterator, image is picked out in an sequatial or random order. **Record system** notes down user's mark for current image, for example, simple label (good or bad), or other comments (To support later). This system is divided to local and global part. *Local record* just takes notes for current image, and after image changes or marker quits, these notes are merged to correponding entry in *global record*, in which marks for al images are stored.

## Statistic to handle subject uncertainty

A way to reduce subject uncertainty in marking image is to repeat in many times or by many people. After that, one image may have more than one labels, for example `{lab0: n0, lab1: n1, ...}`. Question now is how to choose one as final label and whether this choice is statistically significant.

First question is simple, choose the mark with maximum count. But what if there are more one marks with max count. In the situation where priorities of mark is specified by user, then choose the one with highest priority among these marks with max count is an accpetable solution. However, in this priority case, a variant should be thought, where several marks have very approximated counts, not exactly same. Especially mark with higher priority has comparably lower count. Then what is our choice? Statistical significant would be a good approach for it.

Suppose a model, where an experiment, which produces m results in a probability distribution, written as `(p1, p2, .., pm)`, is performed independently in `N` times. If m=2, this is the Bernoulli experiment. And the observation is the count `ni` of ith result, written as `(n1, n2, ..., nm)` (`0 <= ni <= 1`, `N=n1+n2+..+nm`). Now we want to infer the underlying `(pi)`. Back to the mark problem, imaging huge number of people, maybe all people in whole word, the probability `pi` corresponds to fraction of people to mark the image as ith label. In this point, the decision for mark choice could be done based on comparison of `(pi)`.

In Bayesian analysis, information of `(pi)` is coded in the posterior `P((pi) | (ni))`. With uniform prior for `(p1, p2, ..., pm)`, that is probability density `p((pi))=1` in the region `0 <= pi <= 1` and `Sum pi =1`, posterior density could be obtained (`delta(x)` is the delta function):
    ```p_p((pi) | (ni)) = delta(1-Sum pi) Prod pi^ni / I```
where `I` is the evidence, as following (`B(p, q)` is the Beta function):
    ```
    I = Int_0^1 dp1 dp2 ... dpm Prod pi^ni
      = B(n2+1, n1+1) * B(n3+1, n1+n2+2) * ... * B(nm+1, n1+...+n(m-1)+m-1))
      = Prod ni! / (n1+n2+...+nm+m-1)!
    ```
If we focus one `pi`, e.g. `p1`, the margin distribution is:
    ```p_p1(p1 | (ni)) = p1^n1 (1-p1)^(N-n1) / B(N-n1+1, n1+1)```
Same as Bernoulli experiment. Notice that the form is similar as binomial distrition, but they are different, where current random variant is `p1`, instead of `n1`. Furthermore, some statistics could be computed:
    - mean: `mean(p1) = (n1+1)/(N+2)`
    - second moment: `mean(p1^2) = (n1+2)(n1+1)/[(N+3)(N+2)]`
    - max posterior: `mp = n/N`
    - variance: `D(p1) = mean( (p1-mean(p1))^2 ) = (N-n1+1)(n1+1)/[(N+2)^2(N+3)]`

To compare `(pi)`, it is useful to study their ratio. Define `ki = pi / p1` for `i >= 2`. Constrained region is `ki > 0` for all `i >= 2`. Notice that
    ```
    dp1 dp2 ... dpm = Jacob[dpj/dki] dp1 dk2 ... dkm
                    = p1^(m-1) dp1 dk2 ... dkm
    ```
Density function of `(ki)` is
    ```p_k(k2, k3, .., km) = Prod ki^ni / [(1+k2+..+km)^(N+m) * I]```
If we want to compare a pair, like `p1` and `p2`, distribution of `k2 = p2/p1` would be useful. That is the margin density:
    ```p_k2(k2) = k2^n2 / [(1+k2)^(n1+n2+2) * B(n1+1, n2+1)]```
And its `nth` moment is `mean(k2^n) = B(n1-n+1, n2+n+1) / B(n1+1, n2+1)`, which would diverge for `n > n1`. We could also calculate its cumulative distribution function:
    ```P(k2<=x) = 1-B(1/(1+x); n1+1, n2+1) / B(n1+1, n2+1)```
where `B(x; p, q)` is the incomplete Beta function:
    ```B(x; p, q) = Int_0^x dt t^(p-1) (1-t)^(q-1), 0 <= x <= 1```
(another form used in `scipy` is `i(x; p, q) = B(x; p, q) / B(p, q)`)

## strategy to iterate along images

## Examples
