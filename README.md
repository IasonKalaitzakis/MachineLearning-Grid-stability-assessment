Introduction: With the advent of smart grids, bidirectional power flow has introduced stability challenges to formerly unidirectional grids. Decentralized power management exacerbates these challenges, necessitating controlled producer/consumer environments. This project aims to evaluate grid stability based on member parameters.

System Description: The system comprises a four-star node with smart grid "prosumers" forming a star around a central power producer. Grid stability is gauged through frequency, where deviations >0.5Hz pose instability risks and >1Hz can induce blackout.

![image](https://github.com/IasonKalaitzakis/Statistical-modeling-Grid-stability-assessment/assets/31860283/a733e6af-ba89-40ab-bcd5-57fe43990d1a)

Dataset: Derived from simulated data, the dataset includes features like reaction time, power production/consumption, and price elasticity. Dependent variables indicate stability (1) or instability (0) based on the characteristic differential equation's largest root.

Feature extraction and analysis: Correlation maps and Recursive Feature Elimination reveal power features' minimal impact on stability. Elasticity and response time exhibit positive correlations, affirming their significance.

Support Vector Machine: Employing SVM with various kernels, the polynomial kernel achieves the highest accuracy (96%). Blind testing yields promising results with an accuracy of 96% and robust performance metrics.

Gradient Boosting: Gradient Boosting, an ensemble algorithm, outperforms SVM with higher accuracy (98%) and superior confusion matrix results. Despite similar ROC curve performance, GB exhibits computational efficiency.

Conclusions: Both SVM and GB effectively classify stability, with GB slightly outperforming SVM. Power consumption/production minimally influences stability, emphasizing the importance of price elasticity and response delay.
