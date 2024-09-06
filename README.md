## GLMCL_Stance_Detection

![](C:\Users\20171\Desktop\Number_2\第二篇工作图.drawio.png)

##### This is the data and code for our paper GLMCL: a model using the large language model GLM-4 and comparative learning techniques

## Prerequisites

Make sure your local environment has the following installed:



```python
cuda version > 11.0
pytorch>=1.9 
```

## Datastes

We provide the dataset in the [data](https://github.com/Cheng0829/Fuzzy-DDI/blob/master/data) folder.

| Data      | Description                                                  |
| --------- | ------------------------------------------------------------ |
| **SEM16** | The SEM16 dataset (Mohammad et al., 2016) includes tweets from six different topics, including Atheism (A), Hillary Clinton (HC), Legalized Abortion (LA), Climate Change (CC), Donald Trump (DT), and the Feminist Movement (FM). Every data entry is annotated with an opinion towards a particular subject. |
| **Vast**  | The VAST dataset (Allaway & McKeown, 2020) encompasses news commentary information spanning thousands of subjects and is built specifically for zero-shot stance detection (ZSSD). |

## Documentation

```python
  │--README.md
  │--train_model.py
  │--pytorchtools.py
  │--preprocessing.py
  │--modeling.py
  │--model_utils.py
  │--evaluation.py
  │--data_helper.py
  
--data
  │--vast_dev.csv
  |--vast_test.csv
  |--vast_train.csv
  │--A_train.csv
  |--CC_train.csv
  |--DT_train.csv
  |--FM_train.csv
  |--HC_train.csv
  |--LA_train.csv
```

## Train

The experiment used an NVIDIA GeForce RTX 4060 GPU and was trained on Python 3.9, PyTorch 2.0.0, and CUDA 11.8. Although data enhancement will slightly increase the processing time, the total training time for each dataset is controlled to less than 3 hours.

***TODO: More training scripts for easy training will be added soon.***

## Authors

[PQH-hub@github.com](mailto:PQH-hub@github.com)

Email:[pqh@dlmu.edu.cn](mailto:pqh@dlmu.edu.cn)

Site: [GitHub](https://github.com/PQH-hub/stance_detection/blob/main)