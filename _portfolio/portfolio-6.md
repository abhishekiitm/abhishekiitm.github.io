---
title: "Logarithmic Binning for Improved Histogram Plots"
excerpt: "Enhancing histogram visualization by applying log transformation for better binning<br/><img src='/images/thumbnail_log_hist.png'>"
collection: portfolio
---

Log transformation is often discussed while handling skewed data, but its application to create better histograms remains under appreciated.

Here's the difference between two plots, one without using linear bins and using logarithmic bins. 

**Linear bins:**

<img src="/images/plot_lin_hist.png" width="70%"/>

**Logarithmic bins:**

<img src="/images/plot_log_hist.png" width="70%"/>

Plenty of real-world data follow [log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution#Occurrence_and_applications) - population, income, size of media files to name a few.

### Code

<script src="https://gist.github.com/abhishekiitm/d59ab3cbcfaa84112fbdf8ce65369c97.js"></script>