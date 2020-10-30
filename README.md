# Text-Mining

Con text mining si intende il processo di estrazione di informazione dai testi.
È usato nei siti e-commerce (recensioni), social network, aziende e ovunque siano presenti testi non strutturati.
Il problema è che i testi non sono strutturati e per le macchine non ha nessun senso la semantica.
I testi devono essere trasformati in un formato strutturato privo di semantica.
Dal punto di vista matematico non ci sono corrispondenze tra le parole.
All'interno di due testi simili l'unico pattern che si può individuare è il numero di parole comuni tra i testi.

Il problema del text mining è un problema di classificazione. 
Ad esempio si potrebbero voler classificare le recensioni tra positive e negative.

I passi del text mining sono i seguenti:
  1. Costruire un dataset
  2. Labeling delle istanze
  3. Scelta algoritmo adeguato
  4. Addestramento del classificatore
  6. Valutazione delle prestazioni
  
 Il dataset deve essere suddiviso in caratteristiche (features) e classi delle features.
 
 Dopodichè si effettua quella che è la tokenizzazione e vettorizzazione:
 
  i. Tokenizzazione: tutte le frasi del dataset contengono parole. Tutte queste parole (senza ripetizioni) vengono inserite in un vettore. Prendono il nome di token.
  
  ii. Vettorizzazione: si crea una matrice di array. Questa matrice ha tante righe quante sono le istanze nel dataset. Ogni frase del dataset viene confrontata con i token. Se       il token è presente nella frase allora si mette un 1 nel vettore dentro la matrice altrimenti uno 0. La lunghezza dei vettori all'interno della matrice è pari alla             lunghezza del vettore che contiene i token.
  
Per la text-analysis si sfrutta l'algoritmo Bayesiano: probabilità che una certa parola P sia presente in un certo documento D data la classe C.
