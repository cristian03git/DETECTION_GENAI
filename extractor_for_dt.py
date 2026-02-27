#!/usr/bin/env python
# coding: utf-8

# # Per Dataset di Train

# ## Librerie
# E' necessario importare delle librerie, che permettono di effettuare determinate operazioni:
#   1. Importare il modulo `drive`, per accedere a Google Drive da Google Colab, per usarlo come disco;
#   2. Importare la libreria `pandas`, per la manipolazione dei dati tabellari e gestione di **DataFrame**;
#   3. Importare il modulo `files`, per permettere il download del file dal Colab sul computer locale.

# In[ ]:


from google.colab import drive #[1]
import pandas as pd #[2]
from google.colab import files #[3]


# ## Montaggio Drive
# Viene aperta una finestra per fare l'**autenticazione a Google** e accedere ai file da Colab.<br>Monta Drive nella directory `/content/drive`.

# In[ ]:


drive.mount('/content/drive')


# ## Definizione del percorso e dei file da caricare
#   1. Viene specificata la cartella in cui si trovano i file da elaborare, tramite un determinato `base_path`;
#   2. Viene costruita una lista di file, nel formato `.parquet`, da unire; quest'ultimi sono partizioni dello stesso dataset di TRAIN preso in questione (https://huggingface.co/datasets/Jinyan1/COLING_2025_MGT_multingual/viewer/default/train).

# In[ ]:


base_path = "/content/drive/MyDrive/TESI - Classificazione/MIO/partizioni_dt" #[1]
paths = [
    f"{base_path}/train-00000-of-00003.parquet",
    f"{base_path}/train-00001-of-00003.parquet",
    f"{base_path}/train-00002-of-00003.parquet"
] #[2]


# ## Caricamento dei file e Unione
#   1. Viene caricato ogni file `.parquet` in un **DataFrame** separato;
#   2. Vengono uniti tutti i DataFrame in _uno solo_ (`df_full`).<br> `ignore_index=True` fa in modo che l'indice sia rigenerato da zero.

# In[ ]:


print("Caricamento dei file Parquet...") #Messaggio per l'utente
df_list = [pd.read_parquet(path) for path in paths]  #[1]:Lettura di ogni file
df_full = pd.concat(df_list, ignore_index=True)  #[2]:Unione di tutti i DataFrame


# ## Visualizzazione lingue e Input utente
#   1. Estrazione di tutti i valori unici della colonna `"lang"`, ignorando eventuali NaN (_Not a Number_);
#   2. Vengono stampate le lingue presenti nel dataset.
#   3. Viene chiesto all'utente di scegliere una **lingua valida tra quelle elencate**, ripetendo la richiesta fino a quando non viene fornito un input corretto.
# 
# 

# In[ ]:


lingue_disponibili = df_full["lang"].dropna().unique().tolist()  #[1]
print("Lingue disponibili nel dataset:")
print(lingue_disponibili) #[2]

#[3]:
lingua_scelta = ""
while lingua_scelta not in lingue_disponibili:
    lingua_scelta = input("Inserisci una delle lingue disponibili (es. 'it', 'en', ecc.): ").strip()
    if lingua_scelta not in lingue_disponibili:
        print("Lingua non valida. Riprova.")


# ## Filtraggio del Dataframe
#   1. Creazione di un nuovo DataFrame contenente _SOLO LE RIGHE NELLA LINGUA SELEZIONATA_;
#   2. Viene effettuata la stampa del numero di righe filtrate.

# In[ ]:


print(f"Filtraggio della lingua: {lingua_scelta}...")
df_lang = df_full[df_full["lang"] == lingua_scelta]  #[1]
print(f"Totale righe per la lingua '{lingua_scelta}': {len(df_lang)}") #[2]


# ## Salvataggio e Download
#   1. Viene costruito il percorso e il nome del file `.csv` da salvare, includendo la lingua nel nome;
#   2. Salvataggio del Dataframe filtrato in formato CSV su Google Drive. <br> `index=False` permette di evitare il salvataggio della colonna indice;
#   3. Avvio del download del file CSV sul computer dell'utente.

# In[ ]:


output_csv = f"/content/drive/MyDrive/TESI - Classificazione/MIO/{lingua_scelta}_train_full.csv" #[1]
df_lang.to_csv(output_csv, index=False)  #[2]
print(f"File CSV salvato come: {output_csv}") #Messaggio per l'utente
files.download(output_csv) #[3]


# # Per Dataset di Dev
# 

# ## Librerie
# E' necessario importare delle librerie, che permettono di effettuare determinate operazioni:
#   1. Importare il modulo `drive`, per accedere a Google Drive da Google Colab, per usarlo come disco;
#   2. Importare la libreria `pandas`, per la manipolazione dei dati tabellari e gestione di **DataFrame**;
#   3. Importare il modulo `files`, per permettere il download del file dal Colab sul computer locale.

# In[ ]:


from google.colab import drive #[1]
import pandas as pd #[2]
from google.colab import files #[3]


# ## Montaggio Drive
# Viene aperta una finestra per fare l'**autenticazione a Google** e accedere ai file da Colab.<br>Monta Drive nella directory `/content/drive`.

# In[ ]:


drive.mount('/content/drive')


# ## Definizione del percorso e dei file da caricare
#   1. Viene specificata la cartella in cui si trovano i file da elaborare, tramite un determinato `base_path`;
#   2. Viene costruita una lista di file, nel formato `.parquet`, da unire; quest'ultimi sono partizioni dello stesso dataset di DEV preso in questione (https://huggingface.co/datasets/Jinyan1/COLING_2025_MGT_multingual/viewer/default/dev).

# In[ ]:


base_path = "/content/drive/MyDrive/TESI - Classificazione/MIO/partizioni_dt" #[1]
paths = [
    f"{base_path}/dev-00000-of-00001.parquet"
] #[2]


# ## Caricamento dei file e Unione
#   1. Viene caricato ogni file `.parquet` in un **DataFrame** separato;
#   2. Vengono uniti tutti i DataFrame in _uno solo_ (`df_full`).<br> `ignore_index=True` fa in modo che l'indice sia rigenerato da zero.

# In[ ]:


print("Caricamento dei file Parquet...") #Messaggio per l'utente
df_list = [pd.read_parquet(path) for path in paths]  #[1]:Lettura di ogni file
df_full = pd.concat(df_list, ignore_index=True)  #[2]:Unione di tutti i DataFrame


# ## Visualizzazione lingue e Input utente
#   1. Estrazione di tutti i valori unici della colonna `"lang"`, ignorando eventuali NaN (_Not a Number_);
#   2. Vengono stampate le lingue presenti nel dataset.
#   3. Viene chiesto all'utente di scegliere una **lingua valida tra quelle elencate**, ripetendo la richiesta fino a quando non viene fornito un input corretto.

# In[ ]:


lingue_disponibili = df_full["lang"].dropna().unique().tolist()  #[1]
print("Lingue disponibili nel dataset:")
print(lingue_disponibili) #[2]

#[3]:
lingua_scelta = ""
while lingua_scelta not in lingue_disponibili:
    lingua_scelta = input("Inserisci una delle lingue disponibili (es. 'it', 'en', ecc.): ").strip()
    if lingua_scelta not in lingue_disponibili:
        print("Lingua non valida. Riprova.")


# ## Filtraggio del Dataframe
#   1. Creazione di un nuovo DataFrame contenente _SOLO LE RIGHE NELLA LINGUA SELEZIONATA_;
#   2. Viene effettuata la stampa del numero di righe filtrate.

# In[ ]:


print(f"Filtraggio della lingua: {lingua_scelta}...")
df_lang = df_full[df_full["lang"] == lingua_scelta]  #[1]
print(f"Totale righe per la lingua '{lingua_scelta}': {len(df_lang)}") #[2]


# ## Salvataggio e Download
#   1. Viene costruito il percorso e il nome del file `.csv` da salvare, includendo la lingua nel nome;
#   2. Salvataggio del Dataframe filtrato in formato CSV su Google Drive. <br> `index=False` permette di evitare il salvataggio della colonna indice;
#   3. Avvio del download del file CSV sul computer dell'utente.

# In[ ]:


output_csv = f"/content/drive/MyDrive/TESI - Classificazione/MIO/{lingua_scelta}_dev_full.csv" #[1]
df_lang.to_csv(output_csv, index=False)  #[2]
print(f"File CSV salvato come: {output_csv}") #Messaggio per l'utente
files.download(output_csv) #[3]

