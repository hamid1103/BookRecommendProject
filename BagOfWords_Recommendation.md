# Book Recommendations with Bag of Words

The Bag of Words matrix `XTokens` (built in `Leesmaterial.ipynb`) has one row per book and one column per word, so every book is a vector of word counts. To recommend books, we find the books whose vectors are closest to a starting book or student. This is **content-based filtering**, and we measure "closest" with **cosine similarity**.

Cosine similarity ignores how long a text is, so a book with a long description isn't automatically "more similar" to everything else.

## Step 1: "Books like this one" (item-to-item)

```python
from sklearn.metrics.pairwise import cosine_similarity

# Similarity of every book to every other book (n_books x n_books)
sim = cosine_similarity(XTokens)

def recommend_similar(cataloguscode, n=5):
    idx = LD.index[LD["cataloguscode"] == cataloguscode][0]
    scores = sim[idx].copy()
    scores[idx] = -1                      # don't recommend the book itself
    top = scores.argsort()[::-1][:n]
    result = LD.iloc[top][["cataloguscode", "Titel", "genre", "onderwerp"]].copy()
    result["similarity"] = scores[top]
    return result

recommend_similar("CAT-00223")
```

## Step 2: "Books for this student" (using `Recommender_Interacties.csv`)

Build a profile for each student: the average BoW vector of the books they read, weighted by the rating (`beoordeling`) they gave. Then compare that profile with every book.

```python
import numpy as np

def recommend_for_student(student_id, n=5):
    hist = RI[RI["student_id"] == student_id]
    hist = hist[hist["cataloguscode"].isin(LD["cataloguscode"])]
    idxs = LD.index[LD["cataloguscode"].isin(hist["cataloguscode"])]

    weights = hist.set_index("cataloguscode").loc[LD.loc[idxs, "cataloguscode"], "beoordeling"].fillna(3).values
    profile = np.asarray(XTokens[idxs].multiply(weights[:, None]).sum(axis=0))  # 1 x vocab

    scores = cosine_similarity(profile, XTokens).ravel()
    scores[idxs] = -1                     # skip books already read
    top = scores.argsort()[::-1][:n]
    result = LD.iloc[top][["cataloguscode", "Titel", "genre"]].copy()
    result["score"] = scores[top]
    return result

recommend_for_student("S001")
```

The results can then be filtered by `Niveau` or `doelgroep`, so a student only gets books at their reading level.

## Things to fix first

1. **Use the lemmas.** The vectorizer is fit on `Token_List`, but `Lemmatized` also exists. With lemmas, "boeken" and "boek" become the same word, which usually gives better matches:

   ```python
   XTokens = bow.fit_transform(LD["Lemmatized"])
   ```

2. **Cell 51 undoes the normalization.** Both lines read from `LD["Combine"]` instead of `LD["Combine_Normalized"]`, so the lowercasing in cell 50 is lost, and so is the number removal on the line before. It should be:

   ```python
   LD["Combine_Normalized"] = LD["Combine_Normalized"].replace(r"\d+", "", regex=True)
   LD["Combine_Normalized"] = LD["Combine_Normalized"].replace(r"[^\w\s]", "", regex=True)
   ```

3. **Try TF-IDF.** Plain counts give common words like "verhaal" or "fictie" (from the genre column) a lot of weight. `TfidfVectorizer` takes the same arguments as `CountVectorizer` and gives less weight to words that appear in many books. It's often a clear improvement for recommendations, and comparing both is a good addition to the report.

## Evaluation

Hold back some of each student's highly rated books, run the recommender on the rest, and check how often the held-back books show up in the top-n (**precision@k**).
