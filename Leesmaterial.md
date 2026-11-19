# Leesmaterial.ipynb: Documentatie

Dit notebook schoont de leescatalogus op (`Leescatalogus_Recommender_Definitief_leesmateriaal_Jarno.csv`), probeert een paar eerste modellen uit (lineaire regressie, K-Means) en bereidt de tekst voor op een content-based recommender (normalisatie, tokenisatie, lemmatisatie, Bag of Words).

De **Waarom**-regels komen uit de notities en opmerkingen in het notebook. Waar het notebook geen reden geeft, staat _(nog invullen)_.

## Overzicht: problemen per kolom

Uitkomst van de data-analyse (uit `data_analyse.docx`): per kolom het gevonden probleem en de gekozen oplossing. De secties hieronder werken de oplossingen uit.

| Kolom | Probleem | Oplossing |
|---|---|---|
| Titel |  |  |
| Auteur | Er ontbreken 7 auteurs |  |
| Korte omschrijving |  |  |
| Unnamed: 3 | Onnodig, er is geen data aanwezig en het zegt niks over de data | De kolom verwijdert |
| Type materiaal | Verschillen in spelling: boek, Boek, boek-thriller | Met regex alles gewijzigd naar boek |
| Niveau | 6 ontbrekende niveaus | Allen nans naar 2F |
| Thema’s boek | Er ontbreken 134 thema’s |  |
| Materiaal id |  |  |
| Genre | Er ontbreken 26 genres | Alle nans naar fictie zetten |
| Onderwerp | 181 lege onderwerpen |  |
| Trefwoorden | 181 lege onderwerpen |  |
| Taal | Verschillen in NL, NED, nederlands | Alles met regex naar Nederlands gezet |
| Aantal paginas | Een Aantal pagina’s zijn nul of negatieve waarde en 17 ontbreken | alle 0 of negative naar Nan zetten |
| Geschatte leestijd | 9 ontbrekende leestijden | Worden gevuld met de mediaan leestijd van rijen met hetzelfde Type materiaal. |
| Bron url | 234 ontbrekende URL’s, maar boeken hebben geen URL | Doen we niks mee, de ontbrekende zijn boeken en die hebben geen url |
| Doelgroep | 6 ontbrekende doelgroepen | Alle nans naar nt2, zijn artikelen voor nt2 cursisten |  |
| Gegevens status | 8 ontbrekende gegevens status |  |
| Woorden aantal omschrijving |  |  |
| Gem woord lengte |  |  |
| Gemiddelde zinslengte woorden | 8 ontbrekende gemiddelde |  |
| Aandeel lange woorden |  |  |
| Aantal zinnen | Moet aantal zinnen in korte omschrijving worden |  |
| Leescomplexiteit score | 6 ontbrekende leescomplexiteit scores |  |
| Interessecluster | VERHALEN en verhalen zijn anders in format | Met regex alles omgezet naar verhalen |
| Uitgever | Er ontbreken 106 uitgevers |  |
| Publicatie jaar | Er ontbreken 56 jaren |  |
| Beschikbaar formaat | Formaten van online en papier fixen |  |
| Catalogus code |  |  |
| Interne batch code | 69 ontbrekende codes |  |
| Administratieve prioriteit | Er ontbreken 57 prioriteiten |  |
| Doelgroep extra | 72 ontbrekende extra groepen |  |
| Genre kopie | 62 kopieen leeg |  |
| Gemiddelde beoordeling | 100 ontbrekende beoordelingen |  |

> **Let op: afwijkingen tussen deze tabel en de rest van de documentatie**
>
> - **Genre:** de tabel zegt "alle nans naar fictie", maar sectie 2 vult met informatief (artikelen, kranten, tijdschriften, blogposts) en poëzie (dichtbundels).
> - **Niveau:** de tabel zegt "allen nans naar 2f??", het notebook beschrijft dit niet. _(nog invullen)_
> - **Geschatte leestijd:** de tabel noemt de mediaan (366) als beste keuze, dit is in lijn met de code.
> - **Aantal paginas:** de tabel noemt 17 ontbrekende waardes, naast nul of negatief. In sectie 4 worden ≤ 0 waardes `NaN` en vallen ontbrekende rijen weg.
> - **Trefwoorden:** de tabel noemt "181 lege onderwerpen", hetzelfde getal als bij Onderwerp. Dat is waarschijnlijk een kopieerfout.
> - **Aantal zinnen:** de tabel zegt dat dit het aantal zinnen in de korte omschrijving moet worden, in het notebook is dit nog niet gedaan.
> - **Kolommen zonder oplossing:** voor Titel, Auteur, Themas boek, Onderwerp, Trefwoorden, Gegevens status, Uitgever, Publicatie jaar, Interne batch code, Administratieve prioriteit, Doelgroep extra, Genre kopie en Gemiddelde beoordeling is nog geen oplossing ingevuld.

## 1. Inladen en algemene opschoning

```python
LD = pd.read_csv("OriginalDatasets/Leescatalogus_Recommender_Definitief_leesmateriaal_Jarno.csv")
```

- Bij alle tekstkolommen wordt witruimte aan het begin en eind weggehaald en alles wordt lowercase.
- Dubbele rijen worden verwijderd.
- Lege `Type materiaal` wordt gevuld met `"boek"`.
- De kolom `Unnamed: 3` wordt verwijderd als hij bestaat.

> **Waarom (strip + lowercase):** _(nog invullen)_
>
> **Waarom (`Type materiaal` vullen met "boek"):** _(nog invullen)_
>
> **Waarom (`Unnamed: 3` verwijderen):** De kolom bestaat soms wel en soms niet, maar hij is altijd leeg.

## 2. Kolommen fixen

### Taal

Elke waarde in `taal` wordt vervangen door `"Nederlands"`.

> **Waarom:** Handmatig gecontroleerd met `.unique()`: alles is Nederlands, maar het is op verschillende manieren gemarkeerd ("Ned", "Nederlands", "NL").

### Type materiaal

Alle boekvarianten (`boek - thriller`, `boek - roman`, ...) worden `"boek"`. Resultaat: `boek`, `tijdschrift`, `krant (papier)`, `digitale krant`, `online artikel`, `dichtbundel`, `blogpost`.

> **Waarom:** Consistentie: alle variaties van boeken worden één waarde.

### Geschatte leestijd (`geschatte_leestijd_min`)

- Gecontroleerd met een histogram: geen negatieve waardes, dus daar hoeft niks te veranderen.
- 9 rijen zijn leeg. Die worden gevuld met de **mediaan** leestijd van rijen met hetzelfde `Type materiaal`.

> **Waarom:** Groeperen op leesniveau (`Niveau`) is hier niet handig: dan geef je een digitaal artikel 350 minuten. Groeperen op type materiaal geeft realistische waardes.
>
> **Waarom mediaan en niet gemiddelde:** _(nog invullen)_

### Genre en doelgroep

- Plan: lege `genre` vullen (informatief voor artikelen/kranten/tijdschriften/blogposts, poëzie voor dichtbundels) en lege `doelgroep` vullen met `"NT2"`.
- Varianten van `doelgroep` worden samengevoegd (`mbo` → `MBO`, `mbo / nt2` → `MBO/NT2`).

> **Waarom (genre):** De lege genres worden gevuld op basis van `Type materiaal`. Artikelen, kranten, tijdschriften en blogposts zijn informatief en dichtbundels zijn poëzie, dus ze zijn geen fictie. Genre wordt later misschien opnieuw gedaan met NLP.
>
> **Waarom (doelgroep):** De rijen met een lege `doelgroep` zijn gecontroleerd en het zijn kranten/tijdschriften.
>
> **Waarom NT2 voor die rijen:** _(nog invullen)_

### Themas_boek en trefwoorden

- Waardes in `Themas_boek` die beginnen met `https` worden `NaN`.
- `"[naturetoday.com]"` in `trefwoorden` wordt `NaN`.

> **Waarom:** Sommige entries hadden de **bron-URL** van een artikel in `Themas_boek` staan in plaats van thema's.
>
> **Waarom (trefwoorden):** _(nog invullen)_

### Interessecluster

`"VERHALEN"` → `"verhalen"`. Waardes: `verhalen`, `vrije tijd`, `maatschappij`.

> **Waarom:** Consistentie (`verhalen` en `VERHALEN` werden allebei gebruikt).

### Beschikbaar formaat

`online`/`Online` → `Online`, `papier`/`Papier` → `Papier`. Waardes: `pdf`, `digitaal`, `Online`, `Papier`, `NaN`.

> **Waarom:** Consistentie (beide schrijfwijzen werden gebruikt).

## 3. Wat we kunnen voorspellen

Uit de notities in het notebook:

- **Leescomplexiteit score**: de complexiteitsscore van een student voorspellen, zodat boeken met vergelijkbare scores aangeraden kunnen worden.
- **Leesniveau**: _(nog invullen)_
- **Geschatte leestijd**: _(nog invullen)_
- **Doelgroep**: _(nog invullen)_

Misschien:

- Thema's
- Onderwerp
- Misschien die twee samenvoegen (dimensionality reduction)

## 4. Lineaire regressie: leestijd → aantal pagina's

- `aantal_paginas` ≤ 0 wordt `NaN`, en alleen rijen met een geldig aantal pagina's blijven over (`LD_paginas`).
- `X = geschatte_leestijd_min`, `y = aantal_paginas`, 80/20 train/test-split (`random_state=42`), geschaald met `StandardScaler`.
- `LinearRegression` wordt getraind, en de train- en testpunten worden geplot.

> **Waarom (≤ 0 pagina's):** 0 of een negatief aantal pagina's is onmogelijk, dus dat wordt behandeld als ontbrekend.
>
> **Waarom dit model / deze features:** _(nog invullen)_
>
> **Resultaten / evaluatie:** _(nog invullen)_

## 5. K-Means clustering

- Features: `geschatte_leestijd_min` en `aantal_paginas`, geschaald met `StandardScaler`.
- `k` van 2 tot 8 wordt vergeleken met de elbow-methode (inertia) en de silhouette score.
- Eindmodel: `k = 3`.

> **Waarom deze features:** Dezelfde features als bij de lineaire regressie.
>
> **Waarom schalen:** Anders domineert leestijd (grotere getallen) de afstand.
>
> **Waarom k = 3:** Duidelijke knik in de elbow-plot, en de silhouette score blijft ongeveer gelijk bij hogere k.

Clusterresultaten:

| Cluster | Gem. leestijd (min) | Gem. pagina's | Rijen | Vooral |
|---|---|---|---|---|
| 0 | 312 | 201 | 121 | boeken (107), dichtbundels (11) |
| 1 | 23 | 12 | 50 | online artikelen (32), blogposts (14) |
| 2 | 669 | 266 | 84 | boeken (74), dichtbundels (9) |

> **Interpretatie:** _(nog invullen)_

## 6. Tekstkolom voor NLP

Een nieuwe kolom `Combine` voegt `Korte omschrijving`, `onderwerp`, `genre` en `Themas_boek` samen met spaties (`NaN` → lege string).

> **Waarom:** Alle kolommen behalve `Korte omschrijving` hebben ontbrekende waardes. De relevante informatie zit in `trefwoorden`/`Themas_boek`, `onderwerp` en `Korte omschrijving`. `trefwoorden` en `Themas_boek` bevatten hetzelfde soort data, maar in een ander format (witruimte na de puntkomma), nadat de URL's eruit zijn gehaald. Dus: combineren, en `NaN` invullen met witruimte.
>
> **Waarom `trefwoorden` er niet in zit:** Het heeft precies dezelfde inhoud als `Themas_boek`. Het enige verschil is de witruimte tussen het scheidingsteken en het volgende woord (`woii; onderduik` vs `woii;onderduik`). Als je ze allebei toevoegt, telt elk thema twee keer.

## 7. Interacties × leesmateriaal

`Recommender_Interacties.csv` (`RI`) wordt ingeladen en met een left join op `cataloguscode` aan `LD` gekoppeld (`merged`).

> **Waarom / plan:** Nog niet besloten of we hier iets mee doen.

## 8. NLP

Geplande stappen: normalisatie, tokenisatie, stopwoorden verwijderen, stemming, lemmatisatie, POS-tagging.

### Normalisatie → `Combine_Normalized`

- lowercase
- nummers weg
- leestekens weg
- witruimte aan begin en eind weg
- Nederlandse stopwoorden weg (NLTK `stopwords.words("dutch")`)

> **Waarom:** _(nog invullen)_
>
> **Waarom NLTK voor stopwoorden:** _(nog invullen)_

### Tokenisatie → `Token_List`

`Combine_Normalized` wordt gesplitst op witruimte met `.str.split()`.

> **Waarom:** De split-manier is een suggestie van Jarno (AI).

### Lemmatisatie → `Lemmatized`

spaCy `nl_core_news_sm` (met `parser` en `ner` uitgeschakeld) maakt per rij een lijst met lemma's.

> **Waarom lemmatisatie:** _(nog invullen)_
>
> **Waarom spaCy / dit model:** _(nog invullen)_
>
> **Waarom `parser` en `ner` uit staan:** _(nog invullen)_

Stemming en POS-tagging zitten nog niet in het notebook.

## 9. Bag of Words

`CountVectorizer` op `Token_List`:

- `analyzer=lambda tokens: tokens`: de input is al een lijst met tokens
- `min_df=2`: woorden die in minder dan 2 boeken voorkomen vallen weg
- `max_features=5000`: de 5000 meest voorkomende woorden houden
- `lowercase=False`

Resultaat: `XTokens` met shape **(283, 393)**, dus 283 items en 393 woorden. `df_bow` is dezelfde matrix als DataFrame.

> **Waarom deze instellingen:** _(nog invullen)_

**Volgende stap (uit het notebook):** Om met Bag of Words een recommender te maken, moeten we boeken vinden waarvan de vectoren het dichtst bij elkaar liggen (content-based filtering). De gebruikelijke manier om dat te doen is cosine similarity. Zie [BagOfWords_Recommendation.md](BagOfWords_Recommendation.md).

## Bekende problemen in de huidige code

1. **Genre vullen doet niks (cel 15).** `LD[mask]["genre"] = ...` is chained assignment en schrijft naar een kopie (pandas geeft een `ChainedAssignmentError`). `genre` heeft nog steeds `NaN`. Gebruik `LD.loc[mask, "genre"] = LD.loc[mask, "genre"].fillna("informatief")`.2. **Doelgroep-varianten worden niet samengevoegd (cel 16).** Na cel 2 is alles al lowercase, en de waarde is `mbo/nt2` (zonder spaties), dus het resultaat is nog steeds `MBO`, `mbo/nt2`, `nt2`, `NT2`, `MBO/NT2`.
3. **Vervangingen die niks doen.** `"VERHALEN"` (interessecluster) bestaat na het lowercasen niet meer. `beschikbaar_formaat` heeft nu een mix van hoofdletters (`Online`, `Papier`) en kleine letters (`pdf`, `digitaal`).
4. **Cel 51 maakt de normalisatie ongedaan.** Beide regels lezen uit `LD["Combine"]`, dus het weghalen van nummers gaat verloren. Zie [BagOfWords_Recommendation.md](BagOfWords_Recommendation.md).
5. **Bag of Words gebruikt `Token_List`, niet `Lemmatized`.**
6. **De lineaire regressie wordt niet geëvalueerd** (geen score/MSE op de testset).
