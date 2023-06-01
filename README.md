# XNAP-16 - FMA Music Genre Classification (in PyTorch)
Write here a short summary about your project. The text must include a short introduction and the targeted goals.

Aquest projecte es una recerca desde zero per inetntar classificar archius de musica segons els seu genere a traves de l'implementació de deep learning. Per fer-ho s'utilitzara PyTorch com a llibreria principal i librosa per el tractament d'audio.

## Data - FMA
Per aquest projecte estarem utilitzant les dades de FMA, en concret la seva versió mes reduida anomenada FMA SMALL i les seves metadades FMA METADATA.

Totes les metadades i caracteristiques de les pistes d'audio es poden trobar a [``fma_metadata.zip``](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) (342 MiB).
Els fitxers utilitzats en aquest projecte son els seguents:

* ``tracks.csv``: Metadades per cada track tal com id,title,genres,artist ...

Les pistes d'audio MP3-encoded les podem trobar en el seguent lloc:
* [``fma_small.zip``](https://os.unil.cloud.switch.ch/fma/fma_small.zip) : 8,000 pistes d'audio de 30s, 8 generes balancejats (7.2 GiB)

## Com posar el projecte en funcionament
Per poder reproduir el projecte s'haura de posar en funcionament la seguent infrestructura pas a pas:

1. Has de crear una carpeta anomenada "data" que contindra les metadades de les cançons


## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```



## Contributors
Sergi Tordera - 1608676@uab.cat | 
Txell Monedero                  |
Aina Polo

Xarxes Neuronals i Aprenentatge Profund
Grau de __Data Engineering__, 
UAB, 2023
