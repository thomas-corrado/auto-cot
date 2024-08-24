# ErCheck Auto-CoT: Improved Automatic Chain of Thought Prompting in Large Language Models, Thomas Corrado & Lucy Korte

The information below the horizontal line originates from the AutoCoT repository on https://github.com/amazon-science/auto-cot. The section above the line aims to explain Corrado and Korte's changes to the Auto-CoT paper, why they made those changes, and the improved results these changes generated. 

- ### Motivation for Changes
  - <ins>Lack of Seamless Integration</ins>: The first thing Thomas and Lucy noticed was that the `run_demo.py`, `run_inference.py`, and `utils.py` files did not work seamlessly. It was confusing to run these files together and understand which programs were actually clustering the questions and then sending them to ChatGPT. Over time, Thomas and Lucy concluded that `run_demo.py` would cluster the questions, pull out the center-most question, and then create a demo text file that `run_inference.py` would use to run the Auto-CoT program; `run_inference.py` also utilized the functions and parameters in `utils.py` to perform Auto-CoT. Lucy and Thomas wanted one program that could carry out all of these actions. Integrating `run_demo.py`'s logic to `run_inference.py` was the first change we made. We wanted all users to be able to run our program without headaches and confusion, regardless of their coding background. ChatGPT is a tool for everyone, and it was important to create a project that was accessible. 
  - <ins>Wrong Demonstrations</ins>: Another problem we recognized was that the center-most question retrieval method did not prohibit incorrectly answered questions from being selected for the demonstration text sent to ChatGPT. Questions in the demonstration text answered incorrectly means that ChatGPT has a higher likelihood of also answering the target question incorrectly. Our next change was eliminating this problem while maintaining diversity-based clustering.
  - <ins>Poor Performance</ins>: Few-Shot CoT prompting outperformed Auto-CoT on commmon sense questions, achieving 79.5% accuracy compared to 74.4%. We understood the power of Auto-CoT, and we wanted to maximize its performance. 
- ### Original Auto-CoT Logic Explanation
- ### What is ErCheck Auto-CoT?
  - Sampling and Reasoning Chain Validation
  - Clustering
- ### Results
  - Zero Shot performs with 72.6% accuracy.
  - Few Shot utilizes the same demonstration text from Wei et al., 2022a and performs at 74.8% accuracy. 
  - The original AutoCoT model performs with 75.6% accuracy.  
  - Our improved version of AutoCoT, using ErCheck Auto-CoT with 9 clusters, performs with 84.6% accuracy!
- ### Future Extensions 

-----

# Auto-CoT: Automatic Chain of Thought Prompting in Large Language Models (ICLR 2023)

[![Open Auto-CoT in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amazon-science/auto-cot/blob/main/try_cot_colab.ipynb)

Cheer AI up with the "let's think step by step" prompt? More plz. *Let’s think not just step by step, but also one by one.*

Auto-CoT uses more cheers & diversity to SAVE huge manual efforts in chain of thought prompt design, matching or even exceeding performance of manual design on GPT-3.

Check out our [25-page paper](https://arxiv.org/pdf/2210.03493.pdf) for more information.

![](https://user-images.githubusercontent.com/22279212/194787183-a1f8dff8-a0ad-43a1-827f-819671503860.png)

![](https://user-images.githubusercontent.com/22279212/194787130-d28c9191-588c-41d2-a259-62377f19c934.png)


## Requirements

Python>=3.8
```
pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt
```

## Datasets

Download the datasets from the following:

```
https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/dataset
https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/log
```

## Quick Start

See ```try_cot.ipynb```

## Instructions

Construct Demos:

```
python run_demo.py \
--task multiarith \
--pred_file log/multiarith_zero_shot_cot.log \
--demo_save_dir demos/multiarith
```

Run inference:

```
python run_inference.py \
--dataset multiarith \
--demo_path demos/multiarith \
--output_dir experiment/multiarith
```

## Citing Auto-CoT
```
@inproceedings{zhang2023automatic,
  title={Automatic Chain of Thought Prompting in Large Language Models},
  author={Zhang, Zhuosheng and Zhang, Aston and Li, Mu and Smola, Alex},
  booktitle={The Eleventh International Conference on Learning Representations (ICLR 2023)},
  year={2023}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
