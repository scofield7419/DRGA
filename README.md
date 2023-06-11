# DRGA： deep reinforcement learning guided attention

This repository includes the code of the paper [Optimizing attention for sequence modeling via reinforcement learning](https://ieeexplore.ieee.org/abstract/document/9352534/) published at IEEE TNNLS.

----------


### Requirement

``` bash 
python>=3.6
tensorflow
tflearn
numpy
```

### Datasets

Text classifications:

- Movie Review (MR)
- AGnews
- Subjectivity (SUBJ)
- Stanford Sentiment Treebank (SST)


### Word embedding

prepare the `glove` word embedding at `emb` file:

``` 
glove.6B.100d.txt
``` 

### Running


``` bash 
python core/main.py
```


### Citation

If you use this work, please kindly cite:

```
@article{FeiZRJ22,
  author       = {Hao Fei and
                  Yue Zhang and
                  Yafeng Ren and
                  Donghong Ji},
  title        = {Optimizing Attention for Sequence Modeling via Reinforcement Learning},
  journal      = {{IEEE} Trans. Neural Networks Learn. Syst.},
  volume       = {33},
  number       = {8},
  pages        = {3612--3621},
  year         = {2022},
  url          = {https://doi.org/10.1109/TNNLS.2021.3053633}
}
```


----------


### License

The code is released under Apache License 2.0 for Noncommercial use only. 



