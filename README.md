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
   Cluster alignment with a teacher can effectively incorporate the discriminative clustering structures in both domains for better adaptation.     
   CAT leverages a teacher model to reliably discover the class-conditional structure in the feature space. Then, it forces the features of both the source and the target domains to form discriminative class-conditional clusters and aligns the corresponding clusters across domains.      
   a. Discover the class-conditional structures in feature space and reshape them to be discriminative clusters. Source domain is ok since we have labels, target domain is completed by pseudo labels obtained by ensembled models. Clustering-loss: two features belongs to the same clusters should be close, different class features should be far apart (a hinge loss) SNTG loss -- teacher graph, gathering low-dimensional feature mapping.     
   b. Cluster alignment via conditional feature matching: the source features and target features belong to the same class should be similar, similar to feature matching.     
   c. Combining CAT with marginal distributional alignment, i.e., DANN or RevGrad
   
   Thanks to the classconditional discriminative alignment between the source and target domains, CAT outperforms MCD and VADA.   
   Office31 is more realistic and high-dimensional images, providing a good complement to the digits adaptation task.
   
3. FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation

# Distillation
1. Data Distillation: Towards Omni-Supervised Learning. 

# Fine-grainded recognition
1. Local Temporal Bilinear Pooling for Fine-Grained Action Parsing. https://medium.com/syncedreview/local-temporal-bilinear-pooling-for-fine-grained-action-parsing-24a9bdcc1355


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
16. Can gradient clipping mitigate label noise.
    Informally, clipping controls the dynamics of iterates, thus enhancing the rate of convergence to a local minimum. (Capping the global gradient norm at a specified threshold). 
    From optimisation lens: By ensuring that the gradient norm is not too large, the dynamics of iterates are more well-behaved.
    
    From privacy lens: it prevents any single instance from overly influencing parameter updates.
    
    In this paper, we study gradient clipping from *robustness* lens. Intuitively, clipping the gradient prevents over-confident descent steps.
    
    a. gradient clipping alone does NOT endow label noise robustness.
    
    b. The composite loss-based gradient clippling, a variant that have label noise robustness. ONLY clip the contribution of the *base loss*, being partially Huberised loss.
    
17. CURRICULUM LOSS: ROBUST LEARNING AND GENERALIZATION AGAINST LABEL CORRUPTION
    
18. SIMPLE AND EFFECTIVE REGULARIZATION METHODS FOR TRAINING ON NOISILY LABELED DATA WITH GENERALIZATION GUARANTEE  (many useful ideas in the related works) [code](https://link.zhihu.com/?target=https%3A//drive.google.com/drive/folders/1TDlUuL0I-EzIybjz2pMAgyaYP5F6dq6o)
    
    This paper proposes and analyzes two simple and intuitive regularization methods, the generalization bound is independent of the network size, and the comparable to the bound one can tget when there is no label noise.
    
    a. regularization by the distance between the network parameters to initialization--L2SP. THE ROLE OF OVER-PARAMETRIZATION IN GENERALIZATION OF NEURAL NETWORKS.
    
    b. adding a trainable auxiliary variable to the network output for each training example: tries to fit y_i = f(x_i) + \lambda\*b_i. b_i is a trainable variable. The effectiveness of early stopping indicates that clean labels are somewhat easier to fit than wrong labels; therefore, adding an auxiliary variable could help “absorb” the noise in the labels.
    
    The ability to over-fit the entire training dataset is undesirable for generalization when noisy labels are present. In order to prevent over-fitting to mislabeled data, some form of regularization is necessary, i.e., reduce the *effective size* of the model or *early stopping* (but the validation error may up-down many times, it is trade-off between model precision and training time;)
    
    Gradient Descent with Early Stopping is Provably Robust to Label Noise for Neural Networks: shows under a clustering assumption on data, *gradient descent fits the correct labels before starting to over-fit wrong labels*.
    
    In this line of work (wide neural net and neural tangent kernel), parameters in a wide NN are shown to stay close to their initialization during SGD.
    
19. Learning to Learn from Noisy Labeled Data.     
    A meta-learning update is performed prior to conventional gradient update. Meta-learning based noise-tolerant training to learn from noisy labeled data without human supervision or access to any clean labels. Rather than designing a specific model, we propose a model-agnositc training algorithm.     
    The key idea is *a noise-tolerant model should be able to consistently learn the underlying knowledge from data despite different label noise.*     
    Meta-learning: learning update rule of a learner; finding weight initializations that can be easily fine-tuned or transferred. MAML trains model parameters that can learn well based on a few examples and a few gradient descent steps.     
    
    For SGD, some network parameters are more tolerant to label noise than others.    
    a. for each mini-batch, we generate a variety of synthetic noisy labels on the same images, we update the network parameters using one gradient update.     
    b. Then, we enforce the updated network to give consistent predictions with a teacher model unaffected by the synthetic noise.
    
20. Learning from Noisy Labels with Distillation     
    Use an auxiliary model (trained on small clean dataset as guidance), or use distillation guided by knowledge graph.    
    
21. Learning from noisy large-scale datasets with minimal supervision
22. Learning with Bad Training Data via Iterative Trimmed Loss Minimization
    
    It is useful in poorly curated datasets, irrelevant samples (out-of-distribution), as well as backdoor attacks.
    
    Based on the observations: the evolution of training accuracy is different for clean and bad samples. Repeat (a) selecting samples with lowest current loss. (b) retraining a model on only these samples.

23. Gradient regularization improves accuracy of discrimative models
24. Robust learning with jacobian regularization
25. CleanNet: Transfer learning for scalable image classification training with label noise: use one 'prototype' or 'representative sample' to represent one class, use this prototype to identify if it is noise or not.    
26. Symmetric Cross Entropy for Robust Learning with Noisy Labels     
    DNN with cross entropy exhibits overfitting to noisy labels on some 'easy' classes, while suffers on some other 'hard' classes. **class-biased: some classes overfitting, some class underfitting**. In addition, a low test accuracy (underfitting) on hard classes is a barrier to high overall accuracy (poor performance is not only caused by overfitting).       
    Dimensionality-driven: DNN first learns simple representation via dimensionality compression and then overfit to noise labels via dimensionality expansion.       
    CE by itself is not sufficient for learning of hard classes, especially under the noisy label scenario. Underfitting on hard classes is a major cause for overall performance degradation, due to the fact that the accuracy drop caused by overfitting is relatively small.     
    Symmetric Cross Entropy (SCE) = H(q, p) + H(p, q). Use the predictions as labels: Reverse Cross Entropy.
    
27. Robust training with ensemble consensus (sampling trainable examples from a noisy dataset by relying on small-loss criteria might be impractical)

    *DNN can not generalize to __neighborhoods__ of memorized features*, we hypothesize that noisy examples do not consistently incur small losses on the network under a certain perturbation. Training losses of noisy examples would increase by injecting certain perturbation to network parameters, while those of clean examples would not.
    
    Multiple consensus: M networks generated by adding perturbations
    
28. Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization.

    'Decoupling' and 'Co-teaching+' claim that the 'disagreement' strategy is crucial for alleviating the problem of learning with noisy labels.     
    Differences between 'co-teaching' and 'co-training': co-training uses two different views; co-training did not use the memorization property of DNN (the small loss for one network may be large for the other network).   
    
    a. Decoupling: select not consistent samples to update networks. Update by disagreement: S={(x,y): h1(x) != h2(x)}, update h1 with S and update h2 with S. Similar to hard-mining? Only hard mining is not enough.    
    b. Coteaching: use one network to select 'small-loss' samples to update the other network, and vice versa.    
    c. Coteaching+: 'Coteaching+Decoupling' since coteaching will make two prediction closer with the epochs increasing. (How does Disagreement Help Generalization against Label Corruption?)   

29. Using Pre-Training Can Improve Model Robustness and Uncertainty    
    Pre-training may not improve performance on traditional classification metrics, it improves model robustness and uncertainty estimates. Some researchers claims that the pre-training only imporves wall-clock time.    
    a. Pre-training does tremendously improve the model's adversarial robustness.     
    b. Training longer on a corrupted dataset leads to model deterioration, while for pre-trained model, this is not the case.    
    c. Pre-training can reduce overfitting in case of model calibration, rather than model accuracy.     
    
    Uncertainty estimates need to be useful for detecting out-of-distribution samples.    

30. Label-Noise Robust Generative Adversarial Networks. Noisy learning in the generative model.   
31. Robustness of Conditional GANs to Noisy Labels    


    
    
