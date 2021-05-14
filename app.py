import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np
import re
from IPython.display import Image, HTML
from PIL import Image
import itertools
import os
import functions


#Reading the Datas of Foods
breakfast = pd.read_csv("Data/Breakfast.csv")
lunch = pd.read_csv("Data/Lunch.csv")
dinner = pd.read_csv("Data/Dinner.csv")

#Applying the function on the necessary columns of the "Breakfast" dataframe
breakfast['Carbs'] = breakfast['Carbs'].apply(lambda x: functions.find_number(x))
breakfast['Protein'] = breakfast['Protein'].apply(lambda x: functions.find_number(x))
breakfast['Fat'] = breakfast['Fat'].apply(lambda x: functions.find_number(x))

#Applying the function on the necessary columns of the "Lunch" dataframe
lunch['Carbs'] = lunch['Carbs'].apply(lambda x: functions.find_number(x))
lunch['Protein'] = lunch['Protein'].apply(lambda x: functions.find_number(x))
lunch['Fat'] = lunch['Fat'].apply(lambda x: functions.find_number(x))

#Applying the function on the necessary columns of the "Dinner" dataframe
dinner['Carbs'] = dinner['Carbs'].apply(lambda x: functions.find_number(x))
dinner['Protein'] = dinner['Protein'].apply(lambda x: functions.find_number(x))
dinner['Fat'] = dinner['Fat'].apply(lambda x: functions.find_number(x))

#Concatenate the 3 dataframes (breakfast, lunch, dinner)
frames = [breakfast, lunch, dinner]
df2 = pd.concat(frames)

#Sort the Dataframe based on column "Calories" on descending order
df2 = df2.sort_values('Calories', ascending= True).reset_index()

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['Ingredients'] = df2['Ingredients'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['Ingredients'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


navigation = st.sidebar.radio('🥗 NAVIGATION', ('Home 🏠', "Diet Sample Recommendation 🍝", "Top 10 Similiar Meals 🍛", "Meal Info 🍰"), 2)

if navigation == "Home 🏠":
    ("""# Welcome To Diet Recommendation System ! 🍏""")
    video_file = open("vid/video3.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)
    ("""## Play for Good Mood  🎶🎵 """)
    #For fun :)
    video_file = open("vid/video.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)
elif navigation == "Diet Sample Recommendation 🍝":
    ("""# Diet Sample Recommendation 🥗""")
    #For fun :)
    video_file = open("vid/video2.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.write("Please input your parameters to get started !")
    a = functions.bmr_calculator()
    b = functions.caloric_amount_daily(a)
    functions.macro_calculator(b)
    functions.purpose(b, breakfast, lunch, dinner)
elif navigation == "Top 10 Similiar Meals 🍛":
    ("""# Top 10 Similiar Meals""")
    functions.get_recommendations(df2, cosine_sim)
elif navigation == "Meal Info 🍰":
    ("""# Get Info About Your Preferred Meal !""")
    functions.exploreMeal(df2)


