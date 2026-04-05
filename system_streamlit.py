import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
# load the data
data = pd.read_csv("Books.csv", low_memory = False)
#data cleaning
data = data.drop_duplicates('Book-Title').reset_index(drop=True)
data.dropna(inplace=True)
data['Year-Of-Publication'] = data['Year-Of-Publication'].astype(int)
# feature engineering
data['Combined_features'] = data['Book-Title']+ " " + data['Book-Author']+ " " + data['Publisher']
data.drop(columns = 'Year-Of-Publication', axis=1, inplace = True)
model_df = data[['Book-Title', 'Book-Author', 'Publisher','Combined_features']]
display_df = data[['ISBN','Book-Title','Book-Author','Image-URL-M']]
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(model_df['Combined_features'])
titles = data['Book-Title'].tolist()

# Model
def recommend(book_title):
    # Normalize titles
    data['Book-Title'] = data['Book-Title'].str.lower()
    data_clean = data.drop_duplicates(subset=['Book-Title']).reset_index(drop=True)

    book_title = book_title.lower()
    titles = data_clean['Book-Title'].tolist()

    # Find closest match
    match = difflib.get_close_matches(book_title, titles, n=1, cutoff=0.6)
    if not match:
        return "Book not found"
    

    matched_title = match[0]
    idx = data_clean[data_clean['Book-Title'] == matched_title].index[0]

    # Compute similarity scores
    similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Enumerate and filter out the matched book itself
    scores = [(i, score) for i, score in enumerate(similarity_scores) if i != idx]

    # Sort and take top 5
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    book_indices = [i for i, _ in sorted_scores]

    # Format titles for display
    data_clean['Book-Title'] = data_clean['Book-Title'].str.title()
    return data_clean.iloc[book_indices][['Book-Title', 'Image-URL-M']]


# STREAMLIT

import base64

# Function to encode the image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load your background image
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with your image
set_background("background.jpg")

st.markdown("""
<style>
div[data-testid="stHorizontalBlock] {
    background-color:rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 15px
    
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    div.stButton > button {
    background-color: purple;
    color:yellow;
    font-weight:bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    border: none;
    animation: pulse 2s infinite;
    transition: 0.3s ease-in-out; 
    }
    @keyframes pulse{
    0% {box-shadow: 0 0 5px #16A34A;
        }
        50% {box-shadow: 0 0 2px
    #16A34A, 0 0 40px #16A34A;    }
        100% { box-shadow: 0 0 5px
    #16A34A;    }
    }
    div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px #16A34A, 0 0 50px #16A34A;
    }
    </style>
    """,
    unsafe_allow_html=True
) 

st.set_page_config(
    page_title = 'Book Recommendation System',
    layout = 'wide'
)
st.title("📚 Vincent Book Recommendation System")
book = st.text_input("📝Enter book title")
st.markdown('---')
if st.button("Recommend"):
    match = difflib.get_close_matches(book, titles, n=1, cutoff=0.6)
    results = recommend(book)
    if results is None:
        st.error("😮Book not found!")
    elif not match:
        st.warning("😮Book not found")
    elif len(book)== 0:
        st.warning("📝Enter Book Title")
    else:
        data['Book-Title'] = data['Book-Title'].str.lower()
        data_clean = data.drop_duplicates(subset=['Book-Title']).reset_index(drop=True)

        book_title = book.lower()
        titles = data_clean['Book-Title'].tolist()
    
        # Find closest match
        match = difflib.get_close_matches(book_title, titles, n=1, cutoff=0.6)
        if not match:
           print("Book not found")
        
    
        matched_title = match[0]
        idx = data_clean[data_clean['Book-Title'] == matched_title].index[0]
    
        # Compute similarity scores
        similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
        # Enumerate and filter out the matched book itself
        scores = [(i, score) for i, score in enumerate(similarity_scores)]
    
        # Sort and take top 5
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[0:1]
        book_indices = [i[0] for i in sorted_scores]
    
        # Format titles for display
        data_clean['Book-Title'] = data_clean['Book-Title'].str.title()
        actual_book = data_clean.iloc[book_indices][['Book-Title', 'Image-URL-M']]
        st.subheader("🎯 Matched Book")
        st.image(actual_book['Image-URL-M'].iloc[0])
        st.write(actual_book['Book-Title'].iloc[0])
        st.markdown("---")
        st.subheader('✨ Recommended Books')

        cols = st.columns(5)

        for i,(_, row) in enumerate(results.iterrows()):
            with cols[i]:
                st.image(row['Image-URL-M'])
                st.write(row['Book-Title'])












