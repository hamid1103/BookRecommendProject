# Leesmaterial.ipynb: Documentation

This notebook cleans the reading catalogue (`Leescatalogus_Recommender_Definitief_leesmateriaal_Jarno.csv`), tries a few first models (linear regression, K-Means), and prepares the text for a content-based recommender (normalization, tokenization, lemmatization, Bag of Words).

The **Why** lines come from the notes and comments in the notebook, translated from Dutch. Where the notebook doesn't say why, the line is left as _(to fill in)_.

## 1. Loading and general cleanup

```python
LD = pd.read_csv("OriginalDatasets/Leescatalogus_Recommender_Definitief_leesmateriaal_Jarno.csv")
```

- All text columns are stripped of leading/trailing whitespace and lowercased.
- Duplicate rows are removed.
- Empty `Type materiaal` is filled with `"boek"`.
- The column `Unnamed: 3` is dropped if it exists.

> **Why (strip + lowercase):** _(to fill in)_
>
> **Why (fill `Type materiaal` with "boek"):** _(to fill in)_
>
> **Why (drop `Unnamed: 3`):** The column sometimes exists and sometimes doesn't, but it is always empty.

## 2. Column fixes

### Taal

Every value in `taal` is replaced with `"Nederlands"`.

> **Why:** Checked by hand with `.unique()`: everything is in Dutch, but it is marked in different ways ("Ned", "Nederlands", "NL").

### Type materiaal

All book variants (`boek - thriller`, `boek - roman`, ...) become `"boek"`. Result: `boek`, `tijdschrift`, `krant (papier)`, `digitale krant`, `online artikel`, `dichtbundel`, `blogpost`.

> **Why:** Consistency: all variations of books become one value.

### Geschatte leestijd (`geschatte_leestijd_min`)

- Checked with a histogram: no negative values, so nothing to change there.
- 9 rows are empty. They are filled with the **median** reading time of rows with the same `Type materiaal`.

> **Why:** Grouping by reading level (`Niveau`) isn't useful here: you'd end up giving a digital article 350 minutes. Grouping by material type gives realistic values.
>
> **Why median instead of mean:** _(to fill in)_ (the note says "average", the code uses `median`)

### Genre and doelgroep

- Plan: fill empty `genre` (informatief for articles/newspapers/magazines/blogposts, poëzie for dichtbundels) and empty `doelgroep` with `"NT2"`.
- Variants of `doelgroep` are merged (`mbo` → `MBO`, `mbo / nt2` → `MBO/NT2`).

> **Why (genre):** The empty genres are filled based on `Type materiaal`. Articles, newspapers, magazines and blogposts are informative, and dichtbundels are poetry, so they're not fictie. Genre might be redone later with NLP.
>
> **Why (doelgroep):** The rows with an empty `doelgroep` were checked and are newspapers/magazines.
>
> **Why NT2 for those rows:** _(to fill in)_

### Themas_boek and trefwoorden

- Values in `Themas_boek` that start with `https` become `NaN`.
- `"[naturetoday.com]"` in `trefwoorden` becomes `NaN`.

> **Why:** Some entries had the **source URL** of an article in `Themas_boek` instead of themes.
>
> **Why (trefwoorden):** _(to fill in)_

### Interessecluster

`"VERHALEN"` → `"verhalen"`. Values: `verhalen`, `vrije tijd`, `maatschappij`.

> **Why:** Consistency (`verhalen` and `VERHALEN` were both used).

### Beschikbaar formaat

`online`/`Online` → `Online`, `papier`/`Papier` → `Papier`. Values: `pdf`, `digitaal`, `Online`, `Papier`, `NaN`.

> **Why:** Consistency (both spellings were used).

## 3. What we could predict

From the notes in the notebook:

- **Leescomplexiteit score**: predict a student's complexity score so books with similar scores can be recommended.
- **Lees niveau**: _(to fill in)_
- **Geschatte leestijd**: _(to fill in)_
- **Doelgroep**: _(to fill in)_

Maybe:

- Themas
- Onderwerp
- Maybe merge those two (dimensionality reduction)

## 4. Linear regression: reading time → number of pages

- `aantal_paginas` ≤ 0 is set to `NaN`, and only rows with a valid page count are kept (`LD_paginas`).
- `X = geschatte_leestijd_min`, `y = aantal_paginas`, 80/20 train/test split (`random_state=42`), scaled with `StandardScaler`.
- `LinearRegression` is fit, and train/test points are plotted.

> **Why (≤ 0 pages):** 0 or a negative number of pages is impossible, so it's treated as missing.
>
> **Why this model / these features:** _(to fill in)_
>
> **Results / evaluation:** _(to fill in)_

## 5. K-Means clustering

- Features: `geschatte_leestijd_min` and `aantal_paginas`, scaled with `StandardScaler`.
- `k` from 2 to 8 is compared with the elbow method (inertia) and the silhouette score.
- Final model: `k = 3`.

> **Why these features:** The same features as the linear regression.
>
> **Why scaling:** Otherwise reading time (bigger numbers) dominates the distance.
>
> **Why k = 3:** Clear bend in the elbow plot, and the silhouette score stays about the same for higher k.

Cluster results:

| Cluster | Avg. reading time (min) | Avg. pages | Rows | Mostly |
|---|---|---|---|---|
| 0 | 312 | 201 | 121 | books (107), dichtbundels (11) |
| 1 | 23 | 12 | 50 | online articles (32), blogposts (14) |
| 2 | 669 | 266 | 84 | books (74), dichtbundels (9) |

> **Interpretation:** _(to fill in)_

## 6. Text column for NLP

A new column `Combine` joins `Korte omschrijving`, `onderwerp`, `genre` and `Themas_boek` with spaces (`NaN` → empty string).

> **Why:** Every column except `Korte omschrijving` has missing values. The relevant information is in `trefwoorden`/`Themas_boek`, `onderwerp` and `Korte omschrijving`. `trefwoorden` and `Themas_boek` hold the same kind of data, but in a different format (whitespace after the semicolon), once the URLs are removed. So: combine them, and fill `NaN` with whitespace.
>
> **Why `trefwoorden` is left out:** It has exactly the same content as `Themas_boek`. The only difference is the whitespace between the separator and the next word (`woii; onderduik` vs `woii;onderduik`). Adding both would count every theme twice.

## 7. Interactions × reading material

`Recommender_Interacties.csv` (`RI`) is loaded and left-joined to `LD` on `cataloguscode` (`merged`).

> **Why / plan:** Not yet decided whether to do anything with this.

## 8. NLP

Planned steps: normalization, tokenization, stopword removal, stemming, lemmatization, POS tagging.

### Normalization → `Combine_Normalized`

- lowercase
- remove numbers
- remove punctuation
- strip leading/trailing whitespace
- remove Dutch stopwords (NLTK `stopwords.words("dutch")`)

> **Why:** _(to fill in)_
>
> **Why NLTK for stopwords:** _(to fill in)_

### Tokenization → `Token_List`

`Combine_Normalized` is split on whitespace with `.str.split()`.

> **Why:** The split approach was suggested by Jarno (AI).

### Lemmatization → `Lemmatized`

spaCy `nl_core_news_sm` (with `parser` and `ner` turned off) makes a list of lemmas for each row.

> **Why lemmatization:** _(to fill in)_
>
> **Why spaCy / this model:** _(to fill in)_
>
> **Why `parser` and `ner` are off:** _(to fill in)_

Stemming and POS tagging are not in the notebook yet.

## 9. Bag of Words

`CountVectorizer` on `Token_List`:

- `analyzer=lambda tokens: tokens`: the input is already a list of tokens
- `min_df=2`: drop words that appear in fewer than 2 books
- `max_features=5000`: keep the 5000 most common words
- `lowercase=False`

Result: `XTokens` with shape **(283, 393)**, meaning 283 items and 393 words. `df_bow` is the same matrix as a DataFrame.

> **Why these settings:** _(to fill in)_

**Next step (from the notebook):** To build a recommender with Bag of Words, we need to find books whose vectors are closest to each other (content-based filtering). The usual way to do that is cosine similarity. See [BagOfWords_Recommendation.md](BagOfWords_Recommendation.md).

## Known issues in the current code

1. **Genre fill does nothing (cell 15).** `LD[mask]["genre"] = ...` is chained assignment and writes to a copy (pandas prints a `ChainedAssignmentError`). `genre` still has `NaN`. Use `LD.loc[mask, "genre"] = LD.loc[mask, "genre"].fillna("informatief")`. The markdown cell above it still says "fictie" and should be updated to informatief/poëzie.
2. **Doelgroep variants aren't merged (cell 16).** Everything is already lowercase after cell 2, and the value is `mbo/nt2` (no spaces), so the result is still `MBO`, `mbo/nt2`, `nt2`, `NT2`, `MBO/NT2`.
3. **No-op replacements.** `"VERHALEN"` (interessecluster) no longer exists after lowercasing. `beschikbaar_formaat` now mixes capitalized (`Online`, `Papier`) and lowercase (`pdf`, `digitaal`) values.
4. **Cell 51 undoes normalization.** Both lines read from `LD["Combine"]`, so number removal is lost. See [BagOfWords_Recommendation.md](BagOfWords_Recommendation.md).
5. **Bag of Words uses `Token_List`, not `Lemmatized`.**
6. **The linear regression isn't evaluated** (no score/MSE on the test set).
