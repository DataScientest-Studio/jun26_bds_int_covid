# Home

Target time: 1 minute

## Script

Hello, and thank you for being here. Our project is about classifying chest X-rays into four categories: COVID-19, Lung Opacity, Normal, and Viral Pneumonia.

The technical goal was to build a strong image-classification model. But the more important question became the one shown here: if a model obtains a high score, can we trust that it learned from the lungs and from medically relevant patterns?

We worked with 21,165 raw X-rays and tested several approaches, starting with simple machine-learning baselines and progressing to convolutional neural networks and transfer learning. Our strongest model reached 93.8 percent test accuracy.

However, we also found evidence of shortcut learning. This means that part of the model's performance came from technical or background information that was correlated with the class, rather than only from lung pathology.

During this presentation, I will first explain the dataset and the risks we discovered. Then I will show the preprocessing pipeline, the models, their results, and the interpretability analysis. I will finish with a live prediction.

It is important to state from the beginning that this is an educational research project. It is not a clinically validated diagnostic tool and should not be used for medical decisions.

## Transition

I will begin with the dataset, because understanding how the data was collected is essential for understanding the model's results.

