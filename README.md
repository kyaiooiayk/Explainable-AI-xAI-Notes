# Explainable AI (xAI) 
*List of notebooks focused on Explainable AI (xAI)*
***

## Verfication vs. validation
- Verification is telling you whether you solved the equation **right**.
- Validation is telling you whether you solved the **right** equation. 
- Rather than focusing on explanations, ML practioner should really concentrate on is performance and whether that performance has been tested in a rigorous, scientific manner. There is a nice parallel of this way of thinking: in medicine is full of drugs and techniques that doctors use because they work, even though no one knows why acetaminophen has been used for a century to treat pain and inflammation, even though we still don’t fully understand the underlying mechanism.
- In other words, what we should care about when it comes to A.I. in the real world is not explanation. It is validation.
***

## Available tools [Ref](https://github.com/EthicalML/awesome-production-machine-learning/blob/master/README.md)
* [Aequitas](https://github.com/dssg/aequitas) - An open-source bias audit toolkit for data scientists, machine learning researchers, and policymakers to audit machine learning models for discrimination and bias, and to make informed and equitable decisions around developing and deploying predictive risk-assessment tools.
* [Alibi](https://github.com/SeldonIO/alibi) - Alibi is an open source Python library aimed at machine learning model inspection and interpretation. The initial focus on the library is on black-box, instance based model explanations.
* [anchor](https://github.com/marcotcr/anchor) - Code for the paper ["High precision model agnostic explanations"](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf), a model-agnostic system that explains the behaviour of complex models with high-precision rules called anchors.
* [captum](https://github.com/pytorch/captum) - model interpretability and understanding library for PyTorch developed by Facebook. It contains general purpose implementations of integrated gradients, saliency maps, smoothgrad, vargrad and others for PyTorch models.
* [casme](https://github.com/kondiz/casme)  - Example of using classifier-agnostic saliency map extraction on ImageNet presented on the paper ["Classifier-agnostic saliency map extraction"](https://arxiv.org/abs/1805.08249).
* [ContrastiveExplanation (Foil Trees)](https://github.com/MarcelRobeer/ContrastiveExplanation) - Python script for model agnostic contrastive/counterfactual explanations for machine learning. Accompanying code for the paper ["Contrastive Explanations with Local Foil Trees"](https://arxiv.org/abs/1806.07470).
* [DeepLIFT](https://github.com/kundajelab/deeplift) - Codebase that contains the methods in the paper ["Learning important features through propagating activation differences"](https://arxiv.org/abs/1704.02685). Here is the [slides](https://docs.google.com/file/d/0B15F_QN41VQXSXRFMzgtS01UOU0/edit?filetype=mspresentation) and the [video](https://vimeo.com/238275076) of the 15 minute talk given at ICML.
* [DeepVis Toolbox](https://github.com/yosinski/deep-visualization-toolbox) - This is the code required to run the Deep Visualization Toolbox, as well as to generate the neuron-by-neuron visualizations using regularized optimization. The toolbox and methods are described casually [here](http://yosinski.com/deepvis) and more formally in this [paper](https://arxiv.org/abs/1506.06579).
* [ELI5](https://github.com/TeamHG-Memex/eli5) - "Explain Like I'm 5" is a Python package which helps to debug machine learning classifiers and explain their predictions.
* [FACETS](https://github.com/PAIR-code/facets) - Facets contains two robust visualizations to aid in understanding and analyzing machine learning datasets. Get a sense of the shape of each feature of your dataset using Facets Overview, or explore individual observations using Facets Dive.
* [Fairness Indicators](https://github.com/tensorflow/fairness-indicators) - The tool supports teams in evaluating, improving, and comparing models for fairness concerns in partnership with the broader Tensorflow toolkit.
* [Fairlearn](https://github.com/fairlearn/fairlearn) - Fairlearn is a python toolkit to assess and mitigate unfairness in machine learning models.
* [FairML](https://github.com/adebayoj/fairml) - FairML is a python toolbox auditing the machine learning models for bias.
* [fairness](https://github.com/algofairness/fairness-comparison) - This repository is meant to facilitate the benchmarking of fairness aware machine learning algorithms based on [this paper](https://arxiv.org/abs/1802.04422).
* [GEBI - Global Explanations for Bias Identification](https://github.com/AgaMiko/GEBI) - An attention-based summarized post-hoc explanations for detection and identification of bias in data. We propose a global explanation and introduce a step-by-step framework on how to detect and test bias. Python package for image data.
* [AI Explainability 360](https://github.com/Trusted-AI/AIX360) - Interpretability and explainability of data and machine learning models including a comprehensive set of algorithms that cover different dimensions of explanations along with proxy explainability metrics.
* [AI Fairness 360](https://github.com/Trusted-AI/AIF360) - A comprehensive set of fairness metrics for datasets and machine learning models, explanations for these metrics, and algorithms to mitigate bias in datasets and models.
* [iNNvestigate](https://github.com/albermax/innvestigate) - An open-source library for analyzing Keras models visually by methods such as [DeepTaylor-Decomposition](https://www.sciencedirect.com/science/article/pii/S0031320316303582), [PatternNet](https://openreview.net/forum?id=Hkn7CBaTW), [Saliency Maps](https://arxiv.org/abs/1312.6034), and [Integrated Gradients](https://arxiv.org/abs/1703.01365).
* [Integrated-Gradients](https://github.com/ankurtaly/Integrated-Gradients) - This repository provides code for implementing integrated gradients for networks with image inputs.
* [InterpretML](https://github.com/interpretml/interpret/) - InterpretML is an open-source package for training interpretable models and explaining blackbox systems.
* [keras-vis](https://github.com/raghakot/keras-vis) -  keras-vis is a high-level toolkit for visualizing and debugging your trained keras neural net models. Currently supported visualizations include: Activation maximization, Saliency maps, Class activation maps.
* [L2X](https://github.com/Jianbo-Lab/L2X) - Code for replicating the experiments in the paper ["Learning to Explain: An Information-Theoretic Perspective on Model Interpretation"](https://arxiv.org/pdf/1802.07814.pdf) at ICML 2018.
* [Lightly](https://github.com/lightly-ai/lightly) - A python framework for self-supervised learning on images. The learned representations can be used to analyze the distribution in unlabeled data and rebalance datasets.
* [Lightwood](https://github.com/mindsdb/lightwood)  -  A Pytorch based framework that breaks down machine learning problems into smaller blocks that can be glued together seamlessly with an objective to build predictive models with one line of code.
* [LIME](https://github.com/marcotcr/lime) - Local Interpretable Model-agnostic Explanations for machine learning models.
* [LOFO Importance](https://github.com/aerdem4/lofo-importance)  - LOFO (Leave One Feature Out) Importance calculates the importances of a set of features based on a metric of choice, for a model of choice, by iteratively removing each feature from the set, and evaluating the performance of the model, with a validation scheme of choice, based on the chosen metric.
* [MindsDB](https://github.com/mindsdb/mindsdb) -   MindsDB is an Explainable AutoML framework for developers. With MindsDB you can build, train and use state of the art ML models in as simple as one line of code.
* [mljar-supervised](https://github.com/mljar/mljar-supervised) - An Automated Machine Learning (AutoML) python package for tabular data. It can handle: Binary Classification, MultiClass Classification and Regression. It provides feature engineering, explanations and markdown reports.
* [NETRON](https://github.com/lutzroeder/netron) - Viewer for neural network, deep learning and machine learning models.
* [pyBreakDown](https://github.com/MI2DataLab/pyBreakDown) - A model agnostic tool for decomposition of predictions from black boxes. Break Down Table shows contributions of every variable to a final prediction.
* [responsibly](https://github.com/ResponsiblyAI/responsibly) - Toolkit for auditing and mitigating bias and fairness of machine learning systems
* [SHAP](https://github.com/slundberg/shap) - SHapley Additive exPlanations is a unified approach to explain the output of any machine learning model.
* [SHAPash](https://github.com/MAIF/shapash) - Shapash is a Python library that provides several types of visualization that display explicit labels that everyone can understand.
* [Skater](https://github.com/datascienceinc/Skater) - Skater is a unified framework to enable Model Interpretation for all forms of model to help one build an Interpretable machine learning system often needed for real world use-cases.
* [WhatIf](https://github.com/pair-code/what-if-tool) - An easy-to-use interface for expanding understanding of a black-box classification or regression ML model.
* [Tensorflow's cleverhans](https://github.com/tensorflow/cleverhans) - An adversarial example library for constructing attacks, building defenses, and benchmarking both. A python library to benchmark system's vulnerability to [adversarial examples](http://karpathy.github.io/2015/03/30/breaking-convnets/).
* [tensorflow's lucid](https://github.com/tensorflow/lucid) - Lucid is a collection of infrastructure and tools for research in neural network interpretability.
* [tensorflow's Model Analysis](https://github.com/tensorflow/model-analysis) - TensorFlow Model Analysis (TFMA) is a library for evaluating TensorFlow models. It allows users to evaluate their models on large amounts of data in a distributed manner, using the same metrics defined in their trainer.
* [themis-ml](https://github.com/cosmicBboy/themis-ml) - themis-ml is a Python library built on top of pandas and sklearn that implements fairness-aware machine learning algorithms.
* [Themis](https://github.com/LASER-UMASS/Themis) - Themis is a testing-based approach for measuring discrimination in a software system.
* [TreeInterpreter](https://github.com/andosa/treeinterpreter) - Package for interpreting scikit-learn's decision tree and random forest predictions. Allows decomposing each prediction into bias and feature contribution components as described [here](http://blog.datadive.net/interpreting-random-forests/).
* [woe](https://github.com/boredbird/woe) - Tools for WoE Transformation mostly used in ScoreCard Model for credit rating
* [XAI - eXplainableAI](https://github.com/EthicalML/xai) - An eXplainability toolbox for machine learning.
***

## A note on the notebook rendering
Each notebook has two versions (all python scripts are unaffected by this):
- One where all the markdown comments are rendered in black& white. These are placed in the folder named `GitHub_MD_rendering` where MD stands for MarkDown.
- One where all the markdown comments are rendered in coloured.
***

## Available tutorials
- [Advanced uses of SHAP values](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes/blob/main/tutorials/GitHub_MD_rendering/Advanced%20uses%20of%20SHAP%20values.ipynb)
- [Common pitfalls in linear model coefficients interpretation](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes/blob/main/tutorials/GitHub_MD_rendering/Common%20pitfalls%20in%20linear%20model%20coefficients%20interpretation.ipynb)
- [Drop_column_feature](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes/blob/main/tutorials/GitHub_MD_rendering/Drop_column_feature.ipynb)
- [Explain NLP Models with LIME](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes/blob/main/tutorials/GitHub_MD_rendering/Explain%20NLP%20Models%20with%20LIME.ipynb)
- [Introduction to permutation importance](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes/blob/main/tutorials/GitHub_MD_rendering/Introduction%20to%20permutation%20importance.ipynb)
- [PDF = Partial Dependence Plot & ICE = Individual Conditional Expectation](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes/blob/main/tutorials/GitHub_MD_rendering/PDF%20%3D%20Partial%20Dependence%20Plot%20%26%20ICE%20%3D%20Individual%20Conditional%20Expectation.ipynb)
- [PDP = Partial Dependence Plots V2](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes/blob/main/tutorials/GitHub_MD_rendering/PDP%20%3D%20Partial%20Dependence%20Plots%20V2.ipynb)
- [Permuatation importance in random forest - cardinality](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes/blob/main/tutorials/GitHub_MD_rendering/Permuatation%20importance%20in%20random%20forest%20-%20cardinality.ipynb)
- [Permutation Importance with Collinearity](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes/blob/main/tutorials/GitHub_MD_rendering/Permutation%20Importance%20with%20Collinearity.ipynb)
- [SHAP - SHapley Additive exPlanations](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes/blob/main/tutorials/GitHub_MD_rendering/SHAP%20-%20SHapley%20Additive%20exPlanations.ipynb)
- [SHapley Additive exPlanation](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes/blob/main/tutorials/GitHub_MD_rendering/SHapley%20Additive%20exPlanation.ipynb)

## References
- [What’s wrong with “explainable A.I.”](https://fortune.com/2022/03/22/ai-explainable-radiology-medicine-crisis-eye-on-ai/)
- [How to build TRUST in Machine Learning, the sane way](https://medium.com/bigabids-dataverse/how-to-build-trust-in-machine-learning-the-sane-way-39d879f22e69)
***
