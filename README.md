Main goal of this project is to provide trainable representation of real-word input data. 
There are two classes implemented so far: `FuzzyLayer` and `DefuzzyLayer`.

# FuzzyLayer

This layer is suitable for cases when you working with data that can be clustered into interpretable  groups e.g. spatial coordinates, multi-function values and etc.

Membership function for this layer have form:

<img id="image" src="http://mathurl.com/ybkw2ohp.png" alt="\mu_{{j}} \left( x,c,a \right) ={e}^{-\sum _{i=0}^{\dim}1/4\,{\frac {
 \left( x_{{i}}-c_{{i}} \right) ^{2}}{{a_{{i}}}^{2}}}}
" style="border: 0; padding: 1ex 2ex 1ex 2ex">

where `x` is an input vector of `dim` length, `c` is centroid of j-th membership function and `a` is vector of scaling factors.


# DefuzzyLayer

`DefuzzyLayer` can be trained to transform output of an model to continuos values. In other words this layer can be interpreted as an ruleset and input to this layer - firing levels for rules. 

<img id="image" src="http://mathurl.com/yabcgzn9.png" alt="d \left( x,r \right) =\sum _{i=0}^{{\it input\_dim}}x_{{i}}r_{{i}}" style="border: 0; padding: 1ex 2ex 1ex 2ex">
