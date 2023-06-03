# XNAP-16 - FMA Music Genre Classification (in PyTorch)

Aquest projecte és una recerca des de zero per intentar classificar arxius de música segons el seu gènere a través de la implementació de deep learning. Per fer-ho s'ha fet servir PyTorch com a llibreria principal i librosa pel tractament d'àudio.

## Data - FMA
Per aquest projecte estarem utilitzant les dades de FMA, en concret la seva versió més reduïda anomenada FMA SMALL i les seves metadades FMA METADATA.

Totes les metadades i característiques de les pistes d'àudio es poden trobar a [''fma_metadata.zip''](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) (342 MiB).
Els fitxers utilitzats en aquest projecte són els següents:

* ``tracks.csv``: Metadades per cada track tal com id, title, genres, artist ...

Les pistes d'audio MP3-encoded les podem trobar en el seguent lloc:
* [``fma_small.zip``](https://os.unil.cloud.switch.ch/fma/fma_small.zip) : 8,000 pistes d'audio de 30s, 8 generes balancejats (7.2 GiB)

## Codi
El projecte conté els següents arxius *ipynb* i *py*:
1. ``main_dataloader.ipynb``: Conté el codi per executar i posar en funcionament el dataloader. Seguir els passos de "com posar el projecte en funcionament" per la seva utilització.
2. ``main_model.py``: Conté les funcions principals per poder realitzar l'entrenament i predicció d'un model. 
3. [``utils_split.py``](https://github.com/mdeff/fma/blob/master/utils.py): Fitxer extret del treball [``GitHub mdeff/fma``](https://github.com/mdeff/fma), conte funcions i classes per tractar els arxius de metadata.
4. ``utils_data.py``: Conté les funcions necessàries de la classe que genera el dataloader. És necessari per poder importar el dataloader com a objecte de *pickle* en altres fitxers.

FALTEN ELS MODELS

## Com posar el projecte en funcionament
**S'ha d'utilitzar un entorn *conda*, i *instalar* les llibreries que es requereixin en els codis proporcioants.**

Per poder reproduir el projecte s'haurà de posar en funcionament la següent infraestructura  pas a pas:

1. S'ha de crear una carpeta anomenada "data" que contindrà les metadades de les cançons. (Executar dins la carpeta):

  * ``curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip``
  * ``unzip fma_metadata.zip`` o ``7z x fma_metadata.zip``
    
2. S'ha de crear una carpeta anomenda ``AUDIO_DIR`` que contindra les pistes de cançons MP3. (Executar dins la carpeta):

  * ``curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip``
  * ``unzip fma_small.zip`` o ``7z x fma_small.zip``
  * ``mv 0* ../``
  * ``mv 1* ../``

3. S'han d'eliminar les següents pistes d'àudio, ja que contenen errors:

  * ``cd AUDIO_DIR/099`` + ``rm 099134.mp3``
  * ``cd AUDIO_DIR/108`` + ``rm 108925.mp3``
  * ``cd AUDIO_DIR/133`` + ``rm 133297.mp3``
  
4. Executar un sol cop el codi del següent fitxer:

  * ``main_dataloader.ipynb``
5. Executar la resta de fitxers que no siguin de Models
6. Executar els fitxers de Models on es farà l'entrenament i es podran realitzar prediccions amb la funció ``predict()``

## Mètode

### Dataloader
La base de dades ([``fma_small.zip``](https://os.unil.cloud.switch.ch/fma/fma_small.zip)) amb la que s’ha treballat conforma d’un total de 8.000 MP3 repartits entre 8 classes balancejades. Les dades es troben guardades en el directori ``AUDIO_DIR`` on dins hi ha 156 carpetes amb diferents mp3. Els noms d’aquestes són números del 000 al 155.
Per generar csv ``data/train_labels.csv``, s’ha fet servir el fitxer ``data/fma_metadata/tracks.csv`` que per la seva lectura s’ha cridat a la funció load de l’arxiu ``utils_split.py``. Aquest csv generat conté els identificadors de les cançons i les seves respectives classes identificades amb un número. Les correspondències són:

- Hip-Hop: 0
- Pop: 1
- Folk: 2
- Experimental: 3
- Rock: 4
- International: 5
- Electronic: 6
- Instrumental:7 

El directori on es troba el split train i test s’anomena ``AUDIO_DIR_SPLIT``. El repartiment que s’ha fet és de 80/20 agafant les cançons de forma aleatòria. Dins de cada directori test o train hi ha 156 carpetes i dins d’aquestes 80% o 20% de les cançons. La ruta per accedir a una cançó és: *“AUDIO_DIR_SPLIT/train/num_carpeta/cançó.mp3”*.

Per la realtzacó del projecte s’ha treballat amb imatges, és per això que tots els MP3 s’han passat a espectrograma i després a imatge. Per la primera transformació s’ha definit la funció ``get_melspectrogram_db_2()`` que reb com a paràmetre la ruta de la cançó. De cada mp3 s’agafen els 10 primers segons i aquests es resamplegen a un sampling rate de 22050 mostres per segon. D’aquesta manera totes les imatges que es generen tenen la mateixa mida. Sobre aquestes es calcula el STFT (Transformada de Fourier de Temps Curt) per obtenir una representació de les diferents amplituds de les frequences de l'audio al llarg del temps. Finalment es calcula l’espectograma mel i es passa a decbels. Es fa ús d’aquest tipus d’espectograma perquè descarta les freqüències més altes, les que els humans no podem escoltar, i així aconseguir la representació de com nosaltres escoltaríem els mp3.

Per passar de les representacions espectrals a imatges s’ha desenvolupat una segona funció ``spec_to_image()``. En aquesta funció es pren una matriu d’espectro com a entrada i es calcula la mitjana i la desviació estàndard de la matriu per normalitzar-la restant-li la mitjana i dividint el resultat entre la desviació estàndard més un petit valor per evitar divisions entre zeros. A continuació, es calculen el valor màxim i mínim de la matriu normalitzada per redimensionar el valor d’aquesta al rang de 0 a 255 per convertir els valors a l’escala de colors d’una imatge de 8 bits.

Un cop definides les funcions pertinents, s’ha generat la classe ``GenerateDataloader()``. Aquesta s’encarrega de fer la crida a les funcions anteriors, generar les imatges i guardar l’etiqueta corresponent al gènere de cadascuna de les cançons utilitzant el dataframe ``train_labels.csv``, on es tenen els gèneres corresponents a cada cançó.

Amb això s’han realitzat dos data loaders, un amb les 8.000 dades i un fent data augmentation amb un total de 12.000 cançons. 

Pel que fa al dataloader amb **data augmentation**, s’ha desenvolupat una nova funció ``get_melspectrogram_db_volum()``, amb la mateixa funció que la ``get_melspectrogram_db()`` per passar els fitxers mp3 a espectogrames mel. En aquest nova funció es passa per paràmetre una nova variable *volum* amb la qual s’indica si es vol pujar, multiplicant la representació espectral mel per 1.5, o baixar el volum, multiplicant la representació espectral mel per 0.5. 

Així mateix, s’ha definit una segona funció ``spec_to_image_noise()`` per passar les representacions espectrals a imatges, molt semblant a la funció ``spec_to_image()``,  però en aquest cas afegint soroll gaussià a la imatge. Aquesta funció genera una matriu de la mateixa dimensió que la imatge amb valors aleatoris extrets d’una distribució gaussiana amb una mitja de 0 i una desviació estàndard de 0.1.

En el cas de voler utilitzar el data augmentation, es genera el dataloader amb la funció ``GenerateDataloader()`` passant per paràmetre el percentatge de dades noves que es vol gener, en aquest cas 40%. D’aquest manera es generen tants files mp3 nous de com s’hagi indicat. En aquest cas, el 50% es veurien modificats canviant el volum i la resta afegint soroll utilitzant les funcions pertinents. 

### Main Model
Per aquest projecte s'ha fer ús del transfer learning degut a la seva eficacia en la classificació d’imatges. El transfer learning consisteix en utilitzar un model preentrenat per extreure característiques significatives de les imatges. Per tant, dins la funció ``initialize_model()`` per inicialitzar el model segons l’arquitectura desitjada i els paràmetres preentrenats. 

Donat que hi ha un nombre limitat de classes i poques dades disponibles per entrenar el model s’ha decidit fer ús del feature extraction. Tanmatex, s’ha generat una funció ``set_parameter_requires_grad()`` per canviar l’atribut *requires_grad* a *False* per cadascun dels paràmetre de la xarxa. Aquest atribut indica si s’han de calcular i emmagatzemar els gradients per als paràmetres durant el backpropagation.

Continuant amb la inicialització del model, es congelen aquests paràmetres amb la funció ``set_parameter_requires_grad()`` mencionada i es modifica l’última capa fully connected perquè coincideixi amb el número de classes. Es defineix aquest capa com a capa lineal que transforma les característiques d’entrada en sortides que representen les probabilitats de les diferents classes fa servir. S’utilitza la funció d’activació softmax que produeix probabilitats normalitzades per classe.

Per últim, s’ha modificat la primera capa convolucional perquè accepti imatges amb un sol canal, ja que aquestes es troben en escala de grisos. Entre altres coses, es determina la mida del filtre (7x7) i  el desplaçament en cada direcció (2x2).

Un cop inicialitzats els models, s’ha desenvolupat una funció ``train_model()``, per entrenar els diferents models implementats passant per paràmetre el model, el dataloader generat anteriorment, la funció de la loss, la funció d’optimització, el nombre d’èpoques, el learning rate i si es vol canviar el learning rate durant l’entrenament. En aquesta funció s’ha anat iterant fins al nombre total d’èpoques. Per cada època s’ha configurat el model com a ``train()`` si estava a la fase d’entrenament o com a ``eval()`` si estava a la fase de validació.

Seguidament, s’iteren les dades i es genera un tensor per les etiquetes referents a cada cançó. Es restableixen els gradients dels paràmetres a 0 per evitar acumulacions que puguin generar actualitzacions incorrectes. En el cas d’estar a la fase d’entrenament s’aplica el càlcul de gradient. S’obtenen els valors de sortida a partir de les dades d’entrada, es calcula la loss comparant la sortida del model amb les etiquetes reals i es guarda al llistat de les pèrdues a la fase pertanyent. En aquest cas amb la funció  la *Cross Entropy Loss* perquè és una tècnica utilitzada normalment per la classificació multiclasse. Per últim, es realitza el backpropagation i s’actualitzem els paràmetres del model utilitzant la funció d’optimització passada per paràmetre. De primer l’algorisme ``RSMProp()``, però finalment s’usa l'algorisme ``Adam()`` perquè donava millors resultats. 

Finalment, es calcula la mitjana de la loss i l’accuracy representatius de l’època, es registren aquestes mètricas a la web wandb i es guarda el model amb millor resultat obtingut.

En el moment que el model mostrava una estabilització  de l’accuracy a mesura que avançava l’entrenament, es va decidir modificar el learning rate a un adaptatiu, en aquest cas el learning rate decay. Per a fer ús d’aquesta tècnica s’ha de passar per paràmetre la variable ``change_lr = True``. D’aquesta manera, a l’inici de cada època es fa un crida a la funció ``lr_decay()``.

Per últim, s’ha definit una funció ``predict()`` que rep com a paràmetres el model ja entrenat, el data loader i la funcó de loss utilitzada. En aquesta es posa el model en mode eval i s’itera sobre les dades de validaton del data loader. Per cada una d’elles es calcula la predicció que retorna el model agafant l'etiqueta amb major probabilitat. Un cop fetes les prediccions, es calculen les mètriques triades com a mètode d'avaluació dels resultats: la matriu de confusió i l’accuracy de cada classe (el valor de la loss també és una mètrica feta servir per l'avaluació, però aquesta és calculada durant l’entrenament). Com a resultat es retorna el groud truth (etiquetes correctes) i les labels predites.


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

En vista del punt de partida, on teniem un **49,55%** d'accuracy i tant el model de train com de test es quedaven estancats i no milloraven, es va decidir ara ya implementar l'estrategia d'utilitzar un **Learning Rate Decay** per inentar que el model convergis milloar al quedarse estancat, i d'aquesta forma intentar arrivar a un accuracy mes elevat. Es va fer us també del dataset que contenia **Data Augementation**, d'aquesta forma intentar que el model contingues mes dades i puguesin extreure mes carecteristiques per classificar millor. (*Es pot veure l'esspecificació d'implementacio del lr decay i Data Augmenation en l'apartat de metode*)

L'estrategia mencionada es va decidir aplicar en dos arquitectures de FeatureExtraction diferents. Com poden ser la ResNet50 que ja veniem utilitzant i una arquitectura EfficentNetB0. La diferencia principal entre elles es que la ResNet50 conte 50 capes, pero extreu mes carecteristiques de les dades d'entrada. En canvi la EfficentNetB0 conte 224 capes, pero es una red mes compacta que no agafa tantes carectersitques de les dades d'entrada.

|   | ResNet50 | EfficientNet |
|---|-------------|-------------|
|Train Loss|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/5f0cc400-f8a1-4525-a52d-df0f8d2f271b) |![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/29dd8bb7-9374-4588-aa1c-91469fe26fe8)|
|Val Loss|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/8343639d-46ee-442d-a95e-9f76dde284d2) |![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/0874f7b5-e5d3-4ca1-8023-32bd9bb13ecf)|

Es pot obersevar com en la loss de la ResNet50 la baixada en el train ja no es tant progresiva com avans i en la validació presenta pics de pujada durant la disminució, aixo podem començar a ser sinmptomes de que el model esta tenint oberfitting, un altre paramatre que ho reforcaria, seria que el train també esta arrivant a una loss mes baixa que la validació. En la loss del model d'EfficientNet veiem una disminusió molt gran en el train, arrivant practicament a 0 i en la validació un augment, aixo es una molt clara indicació d'overfiting ja que el model s'esta adpatant molt a les dades d'entrenament i no es capaç de classificar la validació. Aixo creiem que es pot deure ja que el Model de EfficentNet al recollir menys carecteristiques, pero tenir mes capes, s'adapti massa a aquelles carcerctersitiques d'entrenament. En canvi ResNet al agafar mes carecteristiques apren a generelaitzar millor.

|   | ResNet50 | EfficientNet |
|---|-------------|-------------|
|Train Acc|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/4908d166-059b-4e77-bd4f-f6a93f395844) |![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/be1ffdd4-d54d-4b82-98c6-4c673015284f) |
|Val Acc| ![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/0ace934e-0a2b-45b9-b2ef-d30487dc2ef9)|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/1e8cbfc0-5e92-498a-83d3-16725710be81)
 |

En referencia als accuracys, veiem com en al ResNet no hem obtingut gaire millora respecte al seu model anterior, on ara aconseguim un **50,87%** d'accuracy, senyal de que tot i amb el **Data Augmentation** i el **Learning Rate Decay** el model ja no es capaç de millorar gaire mes. Per altre banda el model d'EfficientNet presenta un oberffiting massium com suposavem. Tot i aixó en vista de que l'acurracy de validació es el mes alt aconseguit fins el moment (un **54,85%**) i que encara teniem marge de millora treien l'overffiting, vem decidir continuar per aquest cami.

#### EfficentNet - Millor Model

En vista de l'overffitng que presentava el model d'EfficientNet i que voliem eliminar, vem decidir implemnetar una tecnica de reulgarització com pot ser el  DropOut (*tal i com en el Learning Rate Decay, la implementació del DropOut esta explicada en l'apartat de Metode*), aquesta tecnica consisteix en apagar algunes de les neurones de la capa congelada amb una probabilitat del 50%, per d'aquesta manera evitar que el model s'adapti molt a les dades d'entrenemnet i aixi poder evitar l'overfitting.

Com a resultat de que l'utilització de **Data Augmentation** no va ser un factor direferncial en la ResNet vem decidir probar dos model nous en EfficientNet treient l'overfting com hem comentat, i on un utilitzaria el dataset amb **Data Augmentation** i l'altre sense. Comentar també que es va seguir mantenint el **Learning Rate Decay**.

*Per diferenicar els models, estarem utilitzant ``DA`` fent referencia a Data Augmentation , respecte si l'estem aplicant o no.*


|   | EfficientNet ``DA`` | EfficientNet |
|---|-------------|-------------|
|Train loss|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/1a6225b2-4110-4a33-943e-293f904caf8f)|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/6dba12fe-3ef8-4c98-842e-e5608b95ba4e)|
|Val loss|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/77b61b7e-2cca-4f33-9f33-9b068dc0e98d)|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/e72ead4f-8f74-4274-8ba3-85d3c4b79402)|

En els dos models observem unes funcions loss molt similars que sembla que encara manetinguin una mica de overfitting, donat que les losses en el train dismineuixen progressivament i arriven a uns valors bastant bons, pero en canvi en en la validació tenen un comportament amb oscilacions i amb resultats no tant bons com en el train, tot i aixó es veu com l'overfiting s'ha reduit molt considerablament si tenim en compte l'anetior model d'EfficientNet vist.

|   | EfficientNet ``DA`` | EfficientNet |
|---|-------------|-------------|
|Train Acc|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/298a6aca-b7d5-4a3c-9dbe-51500fce16d1)|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/a46d675f-d4c6-4f7e-bc07-546965c845f4)|
|Val Acc|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/720ca072-f1df-45cc-bcd6-ffc66cfcc61c)|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/f84bd743-3dd1-4a67-a041-1c0d26385066)|

Respecte a les accuracy dels models, tambe hi trobem un comportament molt similar, amb una mica d'overfitting en el model que utilitza ``DA``, pero practicament insignificant. Tot i aixo el model sense ``DA`` arriva a la maxima accuracy aconseguida, sent aquesta d'un **57,89%**. Aixo es deu a que el ``DA`` no esta tenint un afecte real l'hora de que apendre a classificar millor entre eles generes, com ja haviem vist en la ResNet50. 

 __Confusing Matrix__ EffientNet
 
 ![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/a82be60a-a7e1-4029-b17d-34d3b998ddbd)

Si comparem la confusing matrix obtinguda en el model on no utiltzem ``DA`` amb la primera ResNet50, veiem com ara de forma general es classifiquen molt millor tots els generes, excepte **Pop** que es el genere que segueix tenint mes problemes  i es confon bastant amb **Folck i Rock**, pero no a l'inversa. Veiem també com el genere **Experminetal** ha millorat molt considerablament.

## Conclusions

Com a conclusions de l'article volem comentar 3 temes principals:

1. Ens ha sorpres que la utilització de **Data Augmentation** no hagi sigut un tret diferencial a l'hora de classificar, creiem que pot ser afecte d'un cojunt de raons:
 * Insuficiencia de variabilitat en les dades originals, d'auqesta forma fer un augemnet de dades no afegiria noves carcercteristique que es puguesin aprendre per classificar millor.
 * Al tenir un dataset d'entrenamnet petit, i relacionat amb el punt anterior , l'utilització data augmentation tindria un afecte poc representatiu en les noves carceteristiques que es podrien aprendre.

2. El millor model aconseguit s'esdave de l'utilització de trasnfer leraning fent servir la tecnica de Feature Extraction amb l'arquitectura d'una EfficientNetB0 amb un Learning Rate adaptatiu com pot ser el Decay i la regularització DropOut per evitar l'overfiting. Aconseguint aquest un accuracy final del **57,89%**.

3. Fent una recerca d'altres treballs on també es fes classificació de generes de musica a traves d'espectogrames i fent l'utilització del dataset d'FMA SMALL, s'ha vist que els seus millors resultats son:

 * [``GitHub mdeff/fma``](https://github.com/mdeff/fma/): **17,60%**
 * [``GitHub yijerjer/music_genre_classification``](https://github.com/yijerjer/music_genre_classification/blob/master/README.md): **46,70%**

 Per tant creiem que el nostre article ha arrivat a uns bons resultats.


## Referencies
[1] https://github.com/mdeff/fma/

[2] https://github.com/yijerjer/music_genre_classification/blob/master/README.md

## Contributors
* Sergi Tordera - 1608676@uab.cat
* Txell Monedero - 1599263@uab.cat                
* Aina Polo - 1603334@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de __Data Engineering__, 
UAB, 2023
