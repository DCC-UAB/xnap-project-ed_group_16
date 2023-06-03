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
En aquest apartat anirem explicant els principals models que s'han provat, i les resultats obtinguts sobre els mateixos.

#### ResNet50 - Primera presa de contacte 
Per començar voliem saber quin era el nostre punt de partida, per aixó ens vam besar en una ResNet50 ja que de forma general te un rendiment bo per aquests tipus de classificació.

Com a funció de loss vam utilitzar ``CrossEntropyLoss`` ja que es la mes adequada en classificació multiclass i com a funció d'optimització ``RMSProp``, ja que voliem observar els seus resultats. 

| Train Loss | Val Loss |
| ------------- | ------------- |
|  ![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/2664c425-3e74-47be-ac9a-f3c803efd206) | ![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/68e2fe55-960b-40d1-ba02-fba4c7cd69b7)| 

Respecte a la Loss, podem observar molta semblança entre les dades d'entrenement i test, aixo es indica que el model no presenta overfiting i per tant no s'esta adaptant practicame al conjunt de dades d'entrenament. Tot i que puguem observar algunes variacions en la loss del conjunt de validació, veiem una tendencia general la disminució a mesura que avança l'entrenament. Aixo es una bona senyal i per tant el model esta aprenent ve a generalitzar.

| Train Acc | Val Acc |
|-------------|-------------|
|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/8764274d-251f-4b7a-894d-26c8d6d88a38)  | ![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/62ec8098-c647-488a-b9d8-d7e43ae6bb31)|

Revisant els accuracys confirmem que no hi tenim overfiting. El que si que trobem tant en el train com en la validació, es que hi ha un estancament en la millora del acuracy dels models. Arrivant nomes a un **49,55%** en el model de validació. En vista d'aquestes resultats, com s'ha mencionat en l'aparat i explicació del metode a seguir, s'implentaran diferentes estrategies per intentar abolir-ho i millorar.

 __Confusing Matrix__ 

![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/bf3a8ecc-6803-4a92-a5a7-31d864616c00)

Observant la matriu de confusió respecte el model ResNet , es veu molt clarament com generes com **Folk, HipHop o Rock** tenen presicions molt elevades i son facilment classificables, en canvi d'altres com **Pop o Experimental** son bastant mes conflitius, ja que es confonen amb altres generes i per tant no es classifiquen correctament. Aixo es podria deure a la gran quantitat de diversitat i variacions que es solen trobar en aquests generes. Generes com els primers mencionats mantenen uns patrons molt propis del genere i facilment identificables. 

#### ResNet50 vs EfficentNet

En vista del punt de partida, on teniem un **49,55%** d'accuracy i tant el model de train com de test es quedaven estancats i no milloraven, es va decidir ara ya implementar l'estrategia d'utilitzar un **Learning Rate Decay** per inentar que el model convergis milloar al quedarse estancat, i d'aquesta forma intentar arrivar a un accuracy mes elevat (*Es pot veure l'esspecificació d'implementacio del lr decay en l'apartat de metode*). Es va fer us també del dataset que contenia **Data Augementation**, per intentar que el model al tenir mes dades pugues extreure mes carecteristiques i classificar millor.

L'estrategia mencionada es va decidir aplicar en dos arquitectures de FeatureExtraction diferents. Com poden ser la ResNet50 que ja veniem utilitzant i una arquitectura EfficentNetB0. La diferencia principal entre elles es que la ResNet50 conte 50 capes, pero extreu mes carecteristiques de les dades d'entrada. En canvi la EfficentNetB0 conte 224 capes, pero es una red mes compacta que no agafa tantes carectersitques de les dades d'entrada.

|   | ResNet50 | EfficientNet |
|---|-------------|-------------|
|Train Loss|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/5f0cc400-f8a1-4525-a52d-df0f8d2f271b) |![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/29dd8bb7-9374-4588-aa1c-91469fe26fe8)|
|Val Loss|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/8343639d-46ee-442d-a95e-9f76dde284d2) |![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/0874f7b5-e5d3-4ca1-8023-32bd9bb13ecf)|

Es pot obersevar com en la loss de la ResNet50 la baixada en el train ja no es tant progresiva com avans i en la validació presenta pics de pujada durant la disminució, aixo podem començar a ser sinmptomes de que el model esta tenint oberfitting, un altre paramatre que ho reforcaria, seria que el train també esta arrivant a una loss mes baixa que la validació. En la loss del model d'EfficientNet veiem una disminusió molt gran en el train, arrivant practicament a 0 i en la validació un augment, aixo es una molt clara indicació d'overfiting ja que el model s'esta adpatant molt a les dades d'entrenament i no es capaç de classificar la validació. Aixo creiem que es pot deure ja que el Model de EfficentNet al recollir menys carecteristiques, pero tenir mes capes, s'adapti massa a aquelles carcerctersitiques d

|   | ResNet50 | EfficientNet |
|---|-------------|-------------|
|Train Acc|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/4908d166-059b-4e77-bd4f-f6a93f395844) |![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/be1ffdd4-d54d-4b82-98c6-4c673015284f) |
|Val Acc| ![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/0ace934e-0a2b-45b9-b2ef-d30487dc2ef9)|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/1e8cbfc0-5e92-498a-83d3-16725710be81)
 |


#### EfficentNet - Millor Model


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
