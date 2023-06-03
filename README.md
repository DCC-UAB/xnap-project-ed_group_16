# XNAP-16 - FMA Music Genre Classification (in PyTorch)

Aquest projecte es una recerca desde zero per inetntar classificar archius de musica segons els seu genere a traves de l'implementació de deep learning. Per fer-ho s'utilitzara PyTorch com a llibreria principal i librosa per el tractament d'audio.

## Data - FMA
Per aquest projecte estarem utilitzant les dades de FMA, en concret la seva versió mes reduida anomenada FMA SMALL i les seves metadades FMA METADATA.

Totes les metadades i caracteristiques de les pistes d'audio es poden trobar a [``fma_metadata.zip``](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) (342 MiB).
Els fitxers utilitzats en aquest projecte son els seguents:

* ``tracks.csv``: Metadades per cada track tal com id,title,genres,artist ...

Les pistes d'audio MP3-encoded les podem trobar en el seguent lloc:
* [``fma_small.zip``](https://os.unil.cloud.switch.ch/fma/fma_small.zip) : 8,000 pistes d'audio de 30s, 8 generes balancejats (7.2 GiB)

## Codi
El projecte conte els seguents archius notebook i py:
1. ``main_dataloader.ipynb``: Conte el codi per executar i posar en funcionament el dataloader. Seguir els passos de "com posar el projecte en funcionament" per la seva utilització.
2. ``main_model.py``: Conte les funcions principals per poder realitzar l'entrenament i prediccio d'un model. 
3. [``utils_split.py``](https://github.com/mdeff/fma/blob/master/utils.py): Fitxer extret del treball [``GitHub mdeff/fma``](https://github.com/mdeff/fma), conte funcions i classes per tractar els archius de metadata
4. ``utils_data.py``: Conte les funcions necessaries de la classe que genera el dataloader. Es necessari per poder importar el dataloader com a objecte de pickle en altres fitxers

FALTEN ELS MODELS

## Com posar el projecte en funcionament
**S'ha d'utilitzar un entorn conda, i instalar les llibreries que es requereixin en els codis proporcioants.**

Per poder reproduir el projecte s'haura de posar en funcionament la seguent infrestructura pas a pas:

1. S'ha de crear una carpeta anomenada "data" que contindra les metadades de les cançons. (Executar dins la carpeta):

  * ``curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip``
  * ``unzip fma_metadata.zip`` o ``7z x fma_metadata.zip``
    
2. S'ha de crear una carpeta anomenda "AUDIO_DIR" que contindra les pistes de cançons MP3. (Executar dins la carpeta):

  * ``curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip``
  * ``unzip fma_small.zip`` o ``7z x fma_small.zip``
  * ``mv 0* ../``
  * ``mv 1* ../``

3. S'han d'eliminar les sguents pistes d'audio, ja que contenen errors:

  * ``cd AUDIO_DIR/099`` + ``rm 099134.mp3``
  * ``cd AUDIO_DIR/108`` + ``rm 108925.mp3``
  * ``cd AUDIO_DIR/133`` + ``rm 133297.mp3``
  
4. Executar un sol cop el codi del seguent fitxer  

  * ``main_dataloader.ipynb``
5. Executar la resta de fitxers que no siguin de Models
6. Executar els fitxers de Models on es fara l'entrenament i es podran realitzar prediccions amb la funció predict

## Metode

### Dataloader
Explicar les coses fetes en el dataloader (Com carraguem les dades, spectogrames, imatges, Data augmentation ...)
### Main Model
Explicar com implementem les funcions de train  (dropout, lr decay ) i predict

## Models - Analisis I Resultats

Treure anilisis i resultats rollo com la presentació pero mes extens (taules comparatives etc ROC CURVE?)

## Conclusions

## Treball Futur / Millores

## Referencies
[1] ....



## Contributors
Sergi Tordera - 1608676@uab.cat | 
Txell Monedero                  |
Aina Polo

Xarxes Neuronals i Aprenentatge Profund
Grau de __Data Engineering__, 
UAB, 2023
