# Deep Relevance Ranking

[![Python Version](https://img.shields.io/badge/language-python3.6-blue.svg)](https://www.python.org/downloads/) [![Build Status](https://travis-ci.org/RandolphVI/Deep-Relevance-Ranking.svg?branch=master)](https://travis-ci.org/RandolphVI/Deep-Relevance-Ranking) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/c45aac301b244316830b00b9b0985e3e)](https://www.codacy.com/app/chinawolfman/Deep-Relevance-Ranking?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=RandolphVI/Deep-Relevance-Ranking&amp;utm_campaign=Badge_Grade) [![License](https://img.shields.io/github/license/RandolphVI/Deep-Relevance-Ranking.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Issues](https://img.shields.io/github/issues/RandolphVI/Deep-Relevance-Ranking.svg)](https://github.com/RandolphVI/Deep-Relevance-Ranking/issues)

This repository contains my implementations of some models (e.g., [DRMM](https://arxiv.org/pdf/1711.08611.pdf), [PACRR](https://arxiv.org/pdf/1704.03940.pdf), [ABEL-DRMM](https://arxiv.org/pdf/1809.01682.pdf), etc.) for **deep relevance ranking** in QA & IR.

## Requirements

- Python 3.6
- Tensorflow 1.1 +
- Numpy
- Gensim

## Introduction



## Data

See data format in `data` folder which including the data sample files.

You need to download the dataset(s) you intend to use (**BioASQ** and/or **TREC ROBUST2004**).

```C
cd data
sh get_bioasq_data.sh
sh get_robust04_data.sh
```

### Data Format

This repository can be used in other QA & IR datasets in two ways:
1. Modify your datasets into the same format of the sample.
2. Modify the data preprocess code in `data_helpers.py`.

Anyway, it should depend on what your data and task are.

## Network Structure

### DRMM

![](https://farm8.staticflickr.com/7866/32302266427_0fd25d1d7b_o.png)

References:

- [A deep relevance matching model for ad-hoc retrieval](https://arxiv.org/pdf/1711.08611.pdf)

---

### PACRR

![](https://farm8.staticflickr.com/7839/47244754851_6f419f950f_o.png)

References:

- [PACRR: A position-aware neural IR model for relevance matching](https://arxiv.org/pdf/1704.03940.pdf)

---

### ABEL-DRMM

![]()

### POSIT-DRMM

![]()

References:

- [Deep relevance ranking using enhanced document-query interactions](https://arxiv.org/pdf/1809.01682.pdf)

---


## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Ph.D.

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
