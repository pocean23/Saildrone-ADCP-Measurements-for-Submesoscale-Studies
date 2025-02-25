# Saildrone-ADCP-Measurements-for-Submesoscale-Studies
This repository contains code, and notebooks for the study 'Acoustic Doppler Current Profiler Measurements from Saildrones with Applications to Submesoscale Studies' by [P. Bhuyan](https://pocean23.github.io), [C. B. Rocha](https://cesar-rocha.github.io), [L. Romero](https://airsealab.com/authors/leonel/), and [J. T. Farrar](https://www2.whoi.edu/staff/jfarrar/) (accepted with minor revisions in J.TECH). 

# Abstract
Characterizing submesoscale ocean processes requires high-resolution observations in both space O(1 km) and time O(1 hr).  
In the present analysis, we utilize multiple mobile platforms, such as Saildrones (SDs), to achieve high-resolution synchronous measurements of submesoscale features. However, resolving submesoscale velocity gradients requires velocity accuracies of O(1 cm/s).  

In this study, we first assess Saildrone Acoustic Doppler Current Profiler (ADCP) measurements against a high signal-to-noise ratio shipboard ADCP data, both collected during the Sub-Mesoscale Ocean Dynamics Experiment (S-MODE).  
The results show that the standard 5-minute average Saildrone ADCP along-track velocity difference variability (3 cm/s) is consistent with shipboard ADCP data, confirming its suitability for submesoscale studies.  

However, direct ADCP comparisons between a Saildrone and the R/V *Oceanus* give small biases (~1 cm/s). This bias is associated with spatial inhomogeneities and is unlikely to be caused by surface waves, whose signal is expected to be significant near the surface.  

We also examined the 1 Hz Saildrone ADCP data to determine the best averaging window for high-resolution analyses and found that averaging over 3 minutes (~250 m in space) provides minimum noise.  
We investigate the uncertainty of submesoscale current gradients derived from Saildrone ADCP measurements and find that the velocity gradient at a 2 km scale can be obtained with a `0.1f` uncertainty using four Saildrones.  

The methodologies we developed to ascertain the optimal averaging window are versatile and applicable to other uncrewed surface vehicles (USV) or multiple-ship arrays.

# SIGNIFICANCE STATEMENT

Submesoscale currents, spanning from a few hundred meters to several kilometers and lasting from hours to weeks, play a key role in transferring energy and redistributing water properties, influencing air-sea interactions and shaping marine ecosystems. However, observing these currents is challenging. Saildrone, an innovative platform, collects synchronous oceanic and atmospheric data, including ocean currents, but assessing and refining this data is essential for studying submesoscale processes. In this paper, we assess the ocean current measurements from Saildrone and develop methods to characterize noise in the data. We then use this improved data to estimate the uncertainty in the current measurements and their gradients, helping us determine the reliability of the data for analyzing submesoscale flow characteristics. 

