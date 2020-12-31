1. DA with multiple sources.

2. DA with scarce data or imbalanced data. Consider generative transfer learning. Adaptation on C and G.

3. DA Federate learning. Not transmit data, but transmit the generative model trained on the local machine.  

4. Identify noise samples. Using clustering or its neighbor relationships or select some reliable points. Rather than only using small loss.

5. Update by disagreement for ensemble, for semi-supervised learning or DA. To increase the divergence. Two diverged networks have different abilities to filter different types of error. The optimal peer should be complementary, rather than the consistent. Two students are good at math, one student is good at math and the other one is good at literature.

6. Noisy face or Noisy re-id

7. Distillation: the obtained logits can indicate the relationship between different classes. It is like augmentation on label space.

8. DA with prototypes. Compared with pseudo labels, class prototypes are more accurate and reliable since they are summarized over all the instances. 'Prototype-Assisted Adversarial learning for UDA'. Prototype discriminator. Minimize the variance of the latent representations for each class in the target domain. 'Transferrable Prototypical Networks for Unsupervised Domain Adaptation'

9. Semi-supervised Segmentation. CutMix (needs strong varied perturbations), ClassMix (Segmentation-based data augmentation), s4GAN_MLMT.

10. U square net for better translation results
