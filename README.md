# Few-shot for object detection
1. Incremental Few-Shot Object Detection
2. Few-shot Object Detection via Feature Reweighting
3. Frustratingly Simple Few-Shot Object Detection

# Generative model
1. U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for I2I
2. Real or not real, that is the question: change the output of discriminator to a distribution with a vector

# Multi-label classification
1. Learning a Deep ConvNet for Multi-label Classification with Partial Labels

# Image retrieval
1. NetVLAD with attention: attention can be generated by segmentation, we can use this attention to erase 'something' that have negative effects on retrival (such as person).
2. Unsupervised Part-based Weighting Aggregation of Deep Convolutional Features for Image Retrieval
3. ATTENTION-AWARE AGE-AGNOSTIC VISUAL PLACE RECOGNITION
4. Deep Attentional Structured Representation Learning for Visual Recognition (Some regions provide more discriminative information)


# Domain adaptation
1. GradMix: Multi-source Transfer across Domains and Tasks
2. Cluster Alignment with a Teacher for Unsupervised Domain Adaptation
3. FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation

# Distillation
1. 


# Semi-supervised learning
1. Consistency-based: ReMixMatch (distribution alignment: prevent the model's prediction from collapsing) and FixMatch (new sota: use weakly-augmented samples to produce pseudo-labels whose largest class probability are above a threshold and train the model on strongly-augmented samples)
2. Graph-based (label propagation or use mean-teacher to obtain more reliable feature to construct graph):
   a. Low-shot learning with large-scale diffusion
   b. Transductive Semi-Supervised Deep Learning using Min-Max Features
   c. Label propagation for deep semi-supervised learning
   d. Smooth Neighbors on Teacher Graphs for Semi-supervised Learning
   e. Data-efficient semi-supervised learning by reliable edge mining
3. CoMatch (Contrastive Graph Regularization, jointly learning class probabilities from classification head and image representations from self-supervised learning head): unify consistency regularization, entropy minimization, contrastive learning, graph-based SSL.
   a. Pseudo-labeling heavily rely on the quality of model's prediction, where the prediction mistakes would accumulate.
   b. Self-supervised learning are task-agnostic, the representations may be suboptimal.
   c. Previous graph-based methods built graph on the features which are highly-correlated with the class predictions; due to high dimensionality, Euclidean distance becomes less meaningful (CoMatch considers the relative distance); Computation cost is high.
   d. In CoMatch, the self-supervised branch and classification branch are collaborative with each other, rather than separated parts used in many multi-tasks learning.
   e. Procedure: first, use weakly-augmentations to produce pseudo-graph, then use the pseudo-graph as target to training the embedding graph, which measures the similarity of strongly-augmented samples in the embedding space.

   
# Self-supervised learning
1. Self-supervised Representation Learning using 360° Data
2. Prototypical Contrastive Learning of Unsupervised Representations
3. Unsupervised Learning of View-invariant Action Representations
   
# Noise label learning
1. A Simple yet Effective Baseline for Robust Deep Learning with Noisy Labels 
   Relation to semi-supervised learning: noise label will hurt training while unlabeled data will not
   Relation to deep kernel learning: Minimizing the predictive variance can be explained in the framework of posterior regularization, which lead to the solution to be of some specific form. Training samples are far from the decision boundaries and tend to cluster together.
   Local intrinsic dimensionality (LID) of adversarial example is significantly higher than that of normal data. Ref: Characterizing adversarial subspaces using local intrinsic dimensionality
   
2. Bootstrap-hard:  a self-learning technique that use a convex combination of the given label y and model prediction y_hat as the training target.
3. Forward-correction:  a loss correction method based on the noise transition matrix Tˆ estimated by a pre-trained network.
4. Generalized Cross Entropy: noise-robust loss function.
5. Co-teaching: maintains two networks simultaneously, and cross-trains on instances screened by the “small loss” criteria.
6. MentorNet: a meta-learning model that assigns different weights of training examples based on metalearned curriculum with clean validation dataset.
7. Learning to Reweight: reweight the samples based on clean validation set.
8. AugMix: mixup among k different augmentation.
9. Training deep neural-networks using a noise adaptation layer.
10. Co-Mining: Deep Face Recognition with Noisy Labels. (Mutual learning with two nets)
11. O2U-Net: adjust hyperparameters (learning rate) from Overfitting to Underfitting. The noise labels are usually memorized at the late stage of training as the "hard" samples. Therefore, by tracking the variation of loss of very sample at the different stages, it is possible to detect noise labels.
12. A Novel Self-Supervised Re-labeling: self-supervision to generate robust features; using a parallel network to re-label the noise data.
13. SELF: Learning to filter noisy labels with self-ensemble. Filter noise samples with moving average networks (the average prediction is not consistent with the given label BEFORE over-fitting to the dataset), the filtered samples will be used as unlabeled samples in semi-supervised learning.
14. DivideMix: Learning with noisy labels as semi-supervised learning. Use GMM on per-sample loss to filter noise data as unlabeled data. Next, use MixMatch/co-refinement/co-guessing to train network.
15. Confident Learning: Estimating Uncertainty in Dataset Labels
16. CAN GRADIENT CLIPPING MITIGATE LABEL NOISE
17. CURRICULUM LOSS: ROBUST LEARNING AND GENERALIZATION AGAINST LABEL CORRUPTION
18. SIMPLE AND EFFECTIVE REGULARIZATION METHODS FOR TRAINING ON NOISILY LABELED DATA WITH GENERALIZATION GUARANTEE
19. Using Pre-Training Can Improve Model Robustness and Uncertainty
20. Learning from Noisy Labels with Distillation
21. Learning from noisy large-scale datasets with minimal supervision
22. Learning with Bad Training Data via Iterative Trimmed Loss Minimization
23. Gradient regularization improves accuracy of discrimative models
24. Robust learning with jacobian regularization
25. CleanNet: Transfer learning for scalable image classification training with label noise
