# CONEAS DATASET

This is the Coneas dataset consisting of an enhancement of the [ASOHMO](https://arxiv.org/pdf/2306.02978.pdf) dataset. Here you can find the dataset labeled with argumentative information linked to different types of counter-narratives of three possible different types.
You can also find examples of generated counter-narratives used for manual evaluation of different experimental settings and the script used to finetune models and/or generate the counter-narratives, so experiments can be fully reproduced.
Finally, we also include manual evaluation of generated counter-narratives used to score and evaluate each model's and experiment's performance

## INSTALL REQUIREMENTS

```
pip install requirements.txt
```

## Generating Counter-Narratives

To fine-tune a model or use an existing model to generate counter-narratives you can run the following command:

```
python counter-narratives_generation_and_finetuning.py --dataset DATASET --generation-strategy STRATEGY --language LANGUAGE --use_extra_info EXTRA_INFO --cn_strategy CNSTRATEGY --model_name MODELNAME
```
where:
-  DATASET is either "asohmo", "conan" or "both", meaning that the dataset to be used either for train or generation is ASOHMO, CONAN or a combination of both
-  STRATEGY is either "zeroshot", "fewshot", "finetuned" or "pretraining". Option "pretraining" will not generate counter-narratives but fine-tune a model using the training partition of the dataset and save the model into a subfolder. Option "finetuned" will generate counter-narratives using a finetuned model stored in a local folder. Bot "zeroshot" and "fewshot" options will downlad the model specified by MODELNAME from the huggingface hub.
-  LANGUAGE is either "english" or "multi".
-  EXTRA_INFO is either "collecive", "premises" "all" or "". Option "collective" will add after the hate tweet the information about Collective and Property. Option "premises" will add after the hate tweet, information about Justification and Conclusion. Option "all° will add all the information mentioned before and also the Pivot. Option "" will add no information after the hate tweet. Default value is ""
-  CNSTRATEGY is either "a", "b", "c" or "". Option "a", "b" and "c" will only use for training, counter-narratives of the corresponding type, and for generation, the test set containing tweets with at least one counter-narrative of that type. Option "" will use all counter-narratives and all tweets. Default is ""
-  MODELNAME is any model that can be downloaded from the [huggingface hub](huggingface.co)

## Test Results

We generated counter-narratives for the whole ASOHMO dataset's test partition. All generated counter-narratives are stored in "test_results" folder
