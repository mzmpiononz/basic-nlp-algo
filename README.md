# basic-nlp-algo

Comment creer votre propre algorithme NLP (detecteur de spam, classifieur d'intention, sentimental analysis,...) avec seulement quelques lignes de phrases ?
C'est possible grace à la **semantic hashing method**

# Usage
```
import pickle
from utils import semhash_training, inference_preprocess

# TRAINING
# charger le dataset
filename_train, filename_test = 'data/train.csv', 'data/test.csv'

# liste des intentions
intent_names = ['reservation', 'colis', 'filan-kevitra']

# path du model à creer
model_path = "my_model.pkl"

# Training pour trouver le meilleur model
semhash_training(filename_train, filename_test, intent_names, model_path)

# INFERENCE
# charger le model (à choisir si il y a plusieurs model disponibles)
my_model = pickle.load(open(model_path, 'rb'))

# inference
phrase = 'ho any antsirabe azafady ?'

# prediction
print(intent_names[my_model.predict(inference_preprocess(phrase))[0]])
```

### API
Voici le code du serveur local (Flask) du classifieur d'intention >> [server.py](https://github.com/mzmpiononz/basic-nlp-algo/blob/main/server.py)

### References:
- [know-your-intent](https://github.com/kumar-shridhar/Know-Your-Intent/tree/master)
- [Subword Semantic Hashing for Intent Classification on Small Datasets.](https://arxiv.org/abs/1810.07150)
``` 
@article{shridhar2018subword,
  title={Subword Semantic Hashing for Intent Classification on Small Datasets},
  author={Shridhar, Kumar and Sahu, Amit and Dash, Ayushman and Alonso, Pedro and Pihlgren, Gustav and Pondeknath, Vinay and Simistira, Fotini and Liwicki, Marcus},
  journal={arXiv preprint arXiv:1810.07150},
  year={2018}
}
``` 
