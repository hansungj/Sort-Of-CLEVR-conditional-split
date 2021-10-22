# Sort-Of-CLEVR-conditional-split
![alt text](https://github.com/hansungj/Sort-Of-CLEVR-conditional-split/blob/main/0.png)

This repository was created for the Language and Vision course at University of Stuttgart where compositionality in VQA models was studied: [paper](https://github.com/hansungj/Sort-Of-CLEVR-conditional-split/blob/main/SungjunHan_TermPaper_LanguageandVision.pdf)

The dataset can be generated by 
```
python sort_of_clevr_generator_mac.py --finetune-split-name finetune=10 --finetune-number 10 --train-size 19590
```

In CoGenS, there are 6 question types: 
- Which shape [Colour]?
- [Colour] shape left?
- [Colour] shape up?
- [Colour] closest shape?
- [Colour] furthest shape?
- [Colour] shape count?

# Acknowledgement 
This code is largely based on the implemenation from https://github.com/kimhc6028/relational-networks
