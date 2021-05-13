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

#Function to filter the string extract only the numbers from it, make it float and return it.
def find_number(text):
    num = re.findall('[\d\.]',text)
    return float("".join(num))



#BMR equations:https://doctorholmes.wordpress.com/2011/01/27/losing-weight-with-basal-metabolic-rate-the-mifflin-st-jeor-method/
#A function Determining the basal metabolic rate
def bmr_calculator():
    #Get specific inputs
    global bmr
    name = st.text_input('Please input your name.').lower()
    weightInKgs = int(st.number_input('Please input your weight in kilograms.'))
    heightInCms = int(st.number_input('Please input your height in centimeters.'))
    age = int(st.number_input('Please input your age.'))
    gender = st.selectbox('Please input your gender.', options=['👩 female', '👨 male'])
    if gender == '👩 female':
        bmr = int((10*weightInKgs) + (6.25 * heightInCms) - (5* age) - 161)
        st.text(f'Your Basal Metabolic rate is:  {str(bmr)}.')
        return bmr
    elif gender == '👨 male':
        bmr = int((10*weightInKgs) + (6.25 * heightInCms) - (5* age) + 5)
        st.text(f'⚡ Your Basal Metabolic rate is:  {str(bmr)}.')
        return bmr
    
#A function determining daily caloric needs.
def caloric_amount_daily(bmr):
    #Activity Level Indexes
    activity_level_index = [1.2, 1.375, 1.46, 1.725, 1.8]
    activity_level = st.selectbox('Please select your activity level.', options=["🟢 Sedentary","🟡 Lightly Active(Exercise 1-3 times a week)","🔵 Moderately Active(Exercise 3-4 times a week)","🟠 Very Active(Daily Exercise or Intense exercise 4-5 times a week)","🔴 Extra Active(Intense exercise 6 times a week)"])
    if activity_level == "🟢 Sedentary":
        activity_level_index1 = activity_level_index[0]
    elif activity_level == "🟡 Lightly Active(Exercise 1-3 times a week)":
        activity_level_index1 = activity_level_index[1]
    elif activity_level == "🔵 Moderately Active(Exercise 3-4 times a week)":
        activity_level_index1 = activity_level_index[2]
    elif activity_level == "🟠 Very Active(Daily Exercise or Intense exercise 4-5 times a week)":
        activity_level_index1 = activity_level_index[3]
    elif activity_level == "🔴 Extra Active(Intense exercise 6 times a week)":
        activity_level_index1 = activity_level_index[4]
    caloricNeedsDaily = int(bmr * activity_level_index1)
    st.text(f'To keep your current weight you need {caloricNeedsDaily} calories daily.')
    return caloricNeedsDaily

# A function calculating macros 
def macro_calculator(calories):
    protein_calories = int(0.4 * calories)
    protein_grams = int(protein_calories/4)
    carbs_calories = int(0.4 * calories)
    carbs_grams = int(carbs_calories/4)
    fats_calories = int(0.2 * calories)
    fats_grams = int(fats_calories/9)
    st.text(f'You need {protein_calories} calories from Protein: Which is {protein_grams} grams of protein.\n')
    st.text(f'You need {carbs_calories} calories from Carbs: Which is {carbs_grams} grams of carbs.\n')
    st.text(f'You need {fats_calories} calories from Fats: Which is {fats_grams} grams of fats.\n')
    

#The function determines the purpose of the user and calculates macros for losing/gaining weight
#1kg of fat has 7700 calories
#The function filters the dataframes and randomly recommends a daily diet plan
def purpose(calories, df1, df2, df3):
    #determining the purpose
    lossOrGain = st.selectbox('Please select your purpose', options = ["I want to lose weight" , "I want to gain weight"])
    if lossOrGain == "I want to lose weight":
        #determinig how many kilograms should be lossed
        kgs = float(st.number_input('Please input how many kilograms do you want to lose in a week.'))
        #determining the daily caloric limit, the caloric limits of macros and the corresponiding amounts in grams.
        limit_cals_daily = int(calories - (7700*kgs)/7)
        #If the limit is negative that means the person should not eat at all, and should only burn calories which is unreal.
        if limit_cals_daily >= 0:
            #Calculating calories and grams for Macros
            protein_calories = int(0.4 * limit_cals_daily)
            protein_grams = int(protein_calories/4)
            carbs_calories = int(0.4 * limit_cals_daily)
            carbs_grams = int(carbs_calories/4)
            fats_calories = int(0.2 * limit_cals_daily)
            fats_grams = int(fats_calories/9)
            #Showing the results
            st.write(f'⚡ You need {limit_cals_daily} calories daily. \n')
            st.write(f'⚡ You need {protein_calories} calories from Protein: Which is {protein_grams} grams of protein.\n')
            st.write(f'⚡ You need {carbs_calories} calories from Carbs: Which is {carbs_grams} grams of carbs.\n')
            st.write(f'⚡ You need {fats_calories} calories from Fats: Which is {fats_grams} grams of fats.\n')
            #Creating a list with Overall-Calories and Macro-grams
            l = [limit_cals_daily, fats_grams, carbs_grams, protein_grams]
            # Using Try/Except because the when filtering trough the dataframes they might end up becoming empty for extreme cases
            try:
                #Filtering the dataframes
                df1 = df1[(df1.Calories <= l[0]/3) & (df1.Fat <= l[1]/3) & (df1.Carbs <= l[2]/3) & (df1.Protein <= l[3]/3)]
                df2 = df2[(df2.Calories <= l[0]/3) & (df2.Fat <= l[1]/3) & (df2.Carbs <= l[2]/3) & (df2.Protein <= l[3]/3)]
                df3 = df3[(df3.Calories <= l[0]/3) & (df3.Fat <= l[1]/3) & (df3.Carbs <= l[2]/3) & (df3.Protein <= l[3]/3)]
                #Randomly choosing 1 row from each dataframe
                a = df1.sample()
                b = df2.sample()
                c = df3.sample()
                #Concatenating them into one dataframe
                frames = [a, b, c]
                result = pd.concat(frames)
                #Rearranging the columns
                result = result[["Title", "Image", "Description", "Ingredients", "Calories", "Protein", "Carbs", "Fat", "link"]]
                #setting indexes to default
                result = result.reset_index()
                #Providing Intro to the User
                st.write('We recommend you this diet sample.')
                st.write('1. Breakfast: 7-8 AM ☕🧇')
                st.write('2. Lunch:    12-2PM  🧃🥪')
                st.write('3. Dinner:   6-8 PM  🍷🍝')
                #Returning the content displayed from the dataframe
                return displayData(result)
            #In case of ValueError
            except ValueError:
                st.text(f'We don\'t recommend you to lose {kgs} kilogram(s) in one week.\n It seems unhealthy. Try less! 😢')
        #If the daily amount of Calories is negative it will display the bellow sentence.
        else:
            st.text(f'Your goal is unrealistic. You can\'t lose {kgs} kilogram(s) in one week. Try less! 😢')
    elif lossOrGain == "I want to gain weight":
        #determinig how many kilograms should be gained
        kgs = float(st.number_input('Please input how many kilograms do you want to gain in a week.'))
        # It is normal to gain maximum 2 pounds in a week which is nearly 1 kg
        if kgs < 1:
            #determinig daily the caloric limit, the caloric limits of macros and the corresponiding amounts in grams.
            try:
                gaining_cals_daily = int(calories + (7700*kgs)/7)
                protein_calories = int(0.4 * gaining_cals_daily)
                protein_grams = int(protein_calories/4)
                carbs_calories = int(0.4 * gaining_cals_daily)
                carbs_grams = int(carbs_calories/4)
                fats_calories = int(0.2 * gaining_cals_daily)
                fats_grams = int(fats_calories/9)
                #Displaying the results to the user
                st.write(f'⚡ You need {gaining_cals_daily} calories daily. \n')
                st.write(f'⚡ You need {protein_calories} calories from Protein: Which is {protein_grams} grams of protein.\n')
                st.write(f'⚡ You need {carbs_calories} calories from Carbs: Which is {carbs_grams} grams of carbs.\n')
                st.write(f'⚡ You need {fats_calories} calories from Fats: Which is {fats_grams} grams of fats.\n')
                #Filtering the dataframes
                df1 = df1[(df1.Calories >= gaining_cals_daily /6)]
                df3 = df2[(df2.Calories >= gaining_cals_daily /6)]
                df3 = df3[(df3.Calories >= gaining_cals_daily /6)]
                #Choosing two random rows from each dataframe
                samples1 = [df1.sample() for i in range(0,2)]
                samples2 = [df2.sample() for i in range(0,2)]
                samples3 = [df3.sample() for i in range(0,2)]
                #Putting all the samples in one list
                allSamples = [x for x in itertools.chain(samples1, samples2, samples3)]
                result = pd.concat(allSamples)
                #Rearranging the columns
                result = result[["Title", "Image", "Description", "Ingredients", "Calories", "Protein", "Carbs", "Fat", "link"]]
                #setting index to default
                result = result.reset_index()
                #Providing Intro
                st.write('We recommend you the "Six Meals a Day" diet plan.')
                st.write('1. 1st Breakfast:   7:00 AM  ☕🧇')
                st.write('2. 2nd Breakfast:   10:00 PM ☕🧇')
                st.write('3. 1st Lunch:       1:00 PM  🧃🥪')
                st.write('4. 2nd Lunch:       4:00 PM  🧃🥪')
                st.write('5. 1st Dinner:      7:00 PM  🍷🍝')
                st.write('6. 2nd Dinner:      9:00 PM  🍷🍝')
                #Displaying the content from the dataframe
                return displayData(result)
            except ValueError:
                st.write(f'We don\'t recommend you to lose {kgs} kilogram(s) in one week.\n It is seems unhealthy. Try less!😢')
         #If the user wants to gain more than 1 kg in a week the below sentence will be displayed.
        else:
            st.text(f'Your goal is unrealistic. You can\'t gain {kgs} kilogram(s) in one week. Try less than 1kg! 😢')

# Function to display the data in the Dataframe
def displayData(df):
    for i in range(0, len(df)):
        st.write(str(i+1) + ".  " + str(df["Title"][i]))
        st.image('img1/' + df["Image"][i])
        st.write("💡 Description: " + str(df["Description"][i]))
        st.write("🥗 Ingredients: " + str(df["Ingredients"][i]))
        st.write("⚡ Calories: " + str(df["Calories"][i]))
        st.write("⚡ Protein: " + str(df["Protein"][i]) + " g")
        st.write("⚡ Carbs: " + str(df["Carbs"][i]) + " g")
        st.write("⚡ Fat: " + str(df["Fat"][i])+ " g")
        st.write(df["link"][i])


    
# Function that takes in meal title as input and outputs most similar meals
def get_recommendations(df, cosine_sim):
    str_ = 'You can choose any of the meals from our catalog to see the TOP 10 similiar meals.'
    title = st.selectbox(str_, options = [i for i in df["Title"]])
    # Get the index of the meal that matches the title
    indices = pd.Series(df.index, index=df['Title']).drop_duplicates()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that meal
    sim_scores = list(enumerate(cosine_sim[idx]))
        
    # Sort the meal based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar meals
    sim_scores = sim_scores[1:11]

    # Get the meal indices
    meal_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar meals
    result = df.iloc[meal_indices]
    result = result[["Title", "Image", "Description", "Ingredients", "Calories", "Protein", "Carbs", "Fat", "link"]]
    result = result.reset_index().drop("index", axis = 1)
    st.text("Here are the top 10 most similar meals: ")
    return displayData(result)
    
def ShowMeal(result):
    st.write("❕ Title:  " + str(result[0]))
    st.image('img1/' + result[1])
    st.write("💡 Description:  " + str(result[2]))
    st.write("🥗 Ingredients:  " + str(result[3]))
    st.write("⚡ Calories:  " + str(result[4]))
    st.write("⚡ Protein:  " + str(result[5]) + " g")
    st.write("⚡ Carbs:  " + str(result[6]) + " g")
    st.write("⚡ Fat:  " + str(result[7]) + " g")
    st.write(result[8])

def exploreMeal(df):
    str_ = 'You can choose any of the meals from our catalog to see the information about it.'
    title = st.selectbox(str_, options = [i for i in df["Title"]])
    df = df[["Title", "Image", "Description", "Ingredients", "Calories", "Protein", "Carbs", "Fat", "link"]]
    # Get the index of the meal that matches the title
    indices = pd.Series(df.index, index=df['Title']).drop_duplicates()
    idx = indices[title]
    result = df.iloc[idx]
    return ShowMeal(result)

