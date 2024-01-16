# Bayesian Bandit AB-test

## TL;DR

This repository contains Python code for Bayesian Bandit Testing, an advanced A/B testing method. It dynamically allocates traffic to different versions based on performance, leading to efficient use of traffic and higher conversion rates. The code also includes Bayesian A/B testing, power analysis, and generates various plots for result visualization.

## Introduction

This repository contains Python code for Bayesian Bandit Testing, a method that enhances traditional A/B testing by incorporating Bayesian statistics and multi-armed bandit algorithms. This approach allows for dynamic traffic allocation to different versions based on their performance, leading to more efficient use of traffic and higher overall conversion rates during the test.

### A/B-testing

Traditional A/B testing, also known as hypothesis testing, is a statistical method used to compare two or more versions of a webpage, email, or other user experience to determine which one performs better. It involves showing the different versions to similar visitors at the same time and using statistical analysis to determine which version performs better for a given conversion goal.

However, traditional A/B testing has several limitations. It requires a fixed sample size, which means you need to decide in advance how many visitors will see each version of the user experience. It also requires you to run the test until the end, even if one version is clearly outperforming the others.

### Bayesian A/B-testing

Bayesian A/B testing provides a flexible and intuitive approach to understanding the data. Instead of relying on p-values and confidence intervals, it uses probability distributions to represent the uncertainty about the true conversion rates of the different versions. This allows you to update your understanding of the conversion rates as data comes in, and to make decisions at any time, not just at the end of the test.

### Bayesian Bandit Testing

Bayesian Bandit Testing incorporates the principles of multi-armed bandit algorithms. This approach dynamically allocates traffic to different versions based on their performance, leading to more efficient use of traffic and higher overall conversion rates during the test.

### The code

The Python code in this repository implements these A/B-testing methods for Click Through Rate (CTR) and Cost-per-Click (CpC) optimization. It can be split in three parts

- Classic Hypothesis Testing<br>
  This is based on the chi-2 test from `scipy` as a reference and is implemented in the [hypothesis_test.py](hypothesis_test.py)

- Bayesian A/B-Testing<br>
  The custom Bayesian A/B-test framework uses the Beta and Gamma distributions to capture and quanitify the uncertainty in an A/B-test. This allows for constant evalution and provides a much more intuitive alternative to hypothesis testing. It is implemented in the [bayesian_test.py](bayesian_test.py)

- Bayesian Bandit A/B-Testing<br>
  The Bayesian Bandit dynamically allocates traffic to different versions of a webpage (or other user experiences) based on their performance. This approach allows for more efficient use of traffic and higher overall conversion rates compared to traditional A/B testing. It is implemented in the [bayesian_bandit_test.py](bayesian_bandit_test.py)

### Files

In addition, there are a number of notebooks, which serve as the main code for the various parts:

- `bayesian_ab_testing.ipynb`: This Jupyter notebook contains the main code for the Bayesian A/B Testing. It includes code for plotting power analysis results, winner probabilities, and differences between Bayesian and Chi2-test results. It uses the `BayesianTest` and `Campaign` classes to simulate campaigns and perform Bayesian tests on the data.

- `bayesian_bandit_AA_testing.ipynb` and `bayesian_bandit_AB_testing.ipynb`: These notebooks contain the code for Bayesian Bandit A/A and A/B testing respectively. They use the `BayesianBanditTest` and `Campaign` classes to simulate campaigns and perform Bayesian Bandit tests on the data.

- `campaign.py`: This script contains the `Campaign` class, which simulates a campaign with a given click-through rate (CTR) and impressions. It generates a dataset of impressions and clicks for each campaign.

- `bayesian_test.py`: This script contains the `BayesianTest` class, which performs a Bayesian test on the campaign data. It calculates the posterior distribution of the CTR for each campaign and computes the probability that each campaign has the highest CTR.

- `bayesian_bandit_test.py`: This script contains the `BayesianBanditTest` class, which performs a Bayesian Bandit test on the campaign data. It uses a multi-armed bandit approach to dynamically allocate impressions to the campaigns based on their performance.


## Usage

### Create virtual environment
    python -m venv --system-site-packages venvs/recsys
    source ~/venvs/recsys/bin/activate
    pip install -r requirements.txt

### Run code

To run the code, open the `bayesian_ab_testing.ipynb`, `bayesian_bandit_AA_testing.ipynb`, or `bayesian_bandit_AB_testing.ipynb` file in a Jupyter notebook environment and execute the cells. Make sure to have the `campaign.py`, `bayesian_test.py`, and `bayesian_bandit_test.py` files in the same directory.


## Presentation
You can find the presention under the pdf folder: [Bayesian A/B-testing](pdf/Bayesian%20AB-testing.pdf)



## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Contact
For any quesstions, feel free to reach out to "info at arngren dot com"
