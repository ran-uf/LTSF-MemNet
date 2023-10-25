# MemNet on LTSF
Implementation of functional weights in RKHS for the paper "Universal Recurrent Event Memories for Streaming Data" (IJCNN23).

This code is built on the code base of Informer, which is available: 
https://github.com/zhouhaoyi/Informer2020

## Usage
All datasets are provided in this repository. Run the following cmd for review.
```
cd data
ls
```
To install the requirements
```
pip install -r requirements.txt
```
To run the experiments
```
bash scripts/train.sh
```

## Cite

[MemNet paper](https://arxiv.org/abs/2307.15694):

```
@INPROCEEDINGS{10191277,
  author={Dou, Ran and Principe, Jose},
  booktitle={2023 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Universal Recurrent Event Memories for Streaming Data}, 
  year={2023},
  volume={},
  number={},
  pages={1-10},
  doi={10.1109/IJCNN54540.2023.10191277}}
```
