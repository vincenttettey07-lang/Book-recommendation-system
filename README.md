# Book Recommendation App

A deployed recommendation system that suggests books based on user preferences and similarity signals. Built to demonstrate end-to-end ML engineering: data preparation, recommendation logic, evaluation, and an interactive Streamlit interface.

**Live demo:** [https://book-recommendation-system-majtndx5drmajufn7cp235.streamlit.app/]  

---

## Overview

Most recommendation demos stop at a notebook. This project turns a recommendation workflow into a usable application:

- Users select or search for a book / input preferences
- The system generates personalized recommendations
- Results are ranked and displayed in an interactive UI
- The app is deployed and accessible via a public link

This project was built to explore practical recommendation-system design, including [content-based filtering / collaborative filtering / hybrid approach — update].

---

## Features

- Interactive recommendation interface built with Streamlit
- [Content-based / collaborative / hybrid] recommendation logic
- Preprocessing pipeline for cleaning and preparing book metadata
- Similarity scoring and ranked recommendation output
- Deployed live for easy testing
- Modular code structure for data processing and recommendation logic

---

## How It Works

1. **Data preparation**  
   Book metadata is cleaned, missing values are handled, and relevant features such as title, author, genre, and description are normalized.

2. **Feature representation**  
   [Describe your method: TF-IDF vectors / embeddings / user-item matrix / cosine similarity — update this].

3. **Recommendation generation**  
   The app computes similarity between the selected book/user profile and candidate books, then returns the top-N ranked results.

4. **Interface**  
   Streamlit handles user input, calls the recommendation function, and renders the ranked results.

---

- Evaluated using [precision@k / recall@k / cosine similarity sanity checks / manual relevance review]
- Top-10 recommendations reviewed manually for relevance
- Example: [Precision@10 = X%] or [Qualitative examples included below]

**Example recommendation:**

| Input book | Recommended books |
|---|---|
| [Book A] | [Book B], [Book C], [Book D] |

---

## Tech Stack

- **Language:** Python
- **App framework:** Streamlit
- **Data processing:** pandas, NumPy
- **ML / recommendation:** scikit-learn, [TF-IDF / cosine similarity / other]
- **Deployment:** Streamlit Community Cloud

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip
