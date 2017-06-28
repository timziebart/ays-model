## TODO

2. `aws` to `ays`
3. `degrowth` --> `low growth` (also `dg` --> `lg`)
4. fix `pyviability` examples


## Introduction

The code in this repository was written to run the analysis in [2] and create the pictures therein. It is the analysis of low-complexity socio-economic model with respect to the *TSM framework* [1]. If you find any of the work interesting, please don't hesitate to contact me (Tim.Kittel@pik-potsdam.de).


[1] [http://www.earth-syst-dynam.net/7/21/2016/esd-7-21-2016.html](http://www.earth-syst-dynam.net/7/21/2016/esd-7-21-2016.html)

[2] [https://arxiv.org/abs/1706.04542](https://arxiv.org/abs/1706.04542)


## Setup

The code was tested under Ubuntu Xenial only. If you want to get it running on a different system, please contact me (Tim.Kittel@pik-potsdam.de).

The Code relies on the [`PyViability`-library](https://github.com/timkittel/PyViability), please install it.

The easiest way to install everything is to copy and run the following lines
```
git clone **ays-TODO**
git clone **pyviability-TODO**
cd PyViability
pip install -e .
cd ..
```

### auto-completion

The scripts all provide auto-completion of command line arguments via `argcomplete` (install with `pip install argcomplete`). If you want to use it, please run once
```
activate-global-python-argcomplete --dest=- >> ~/.bash_completion
```
Afterwards you should be able to use [Tab] for auto-completion.


## Running the Code

*Disclaimer I: To understand what the Code does, please (at least) skim through [2] first.*

*Disclaimer II: This code is prepared for scientific work, not a production where security and tight right management is important. The `eval` function is used for simplicity during command line evaluation ...*

In order to follow through the computations done for [2] and create the respective pictures, there are four steps  to take:

1. create the "hair plots" Fig. 3 and Fig. 4 (a)-(c)
2. running the actual analysis of the model,
3. displaying the result as in Fig. 5, and
4. the bifurcation analysis, Fig. 6.

All of the scripts have been written with rather high verbosity in mind so the user is aware what's happening. Still, if you don't find it verbose enough, check out the `-v` flag.


### Creation of the "hair plots"

To create the plots that show the dynamics of the model, Fig. 3 and 4 in [2], use the following lines
```
./aws_show.py default --no-boundary
./aws_show.py default
./aws_show.py degrowth
./aws_show.py energy-transformation
```

### Running the TSM analysis on the model

*Warning: Running the analysis takes a while (on my Laptop ~ 90 Minutes).*

To run the actual TSM analysis, `aws_tsm.py` is used. It has an extensive help, please check `./aws_tsm.py -h`. The analysis in the paper was finally done with:
```
./aws_tsm.py --dg --et --num 200 --boundaries both dg-et-both.out
```
The arguments stand for:
`--dg` : use the *low growth* management option (see [2]),
`--et` : use the *energy transformation* management option (see [2]),
`--num 200` : the grid should have 200 points per dimension, i.e. 200^3 in total,
`--boundaries both` : use both boundaries (see [2]), and
`dg-et-both.out` : file name of the output file.

### Create the result plots

To plot the output, use `aws_tsm_show.py`. If `--regions-style surface` has been used, the surface of the region is estimated using alpha shapes. As this process takes a while, the script will save a cache file to speed up the process the next time it's run. To create the two plots of Fig. 5 in [2], run 
```
./aws_tsm_show.py dg-et-both.out  --analyze-original [240,7e13,0.5e12] 0.005 -r s b --mark red -s shelter-backwater.jpg --regions-style surface
```
and
```
./aws_tsm_show.py dg-et-both.out  --analyze-original [240,7e13,0.5e12] 0.005 -r l --mark red --plot-boundaries-original [[0,400],[3.55e13,9e13],[0.2e12,1e12]] --alpha 0.7 -s current-state-in-lake.jpg --regions-style surface
```
Please check the help message for details on the arguments.

### Bifurcation analysis

the bifurcation analysis is basically not doable without access to an High-Performance Computer. I simply changed the corresponding parameter of the bifurcation analysis using the `--set-parameter` option of `aws_tsm.py` and ran a full analysis for all cases. Then use `aws_tsm_bifurc_show.py` to plot the results.

### More can be done

There are more features for more extensive TSM analysis. Please check the help of the scripts or, if your interest won't be satisified with that, I am very happy (to try) to answer any question (Tim.Kittel@pik-potsdam.de).


