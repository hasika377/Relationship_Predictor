import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from keras.metrics import Precision, Recall


# Title
st.title("💖Relationship Alignment Predictor💖")

# Loading CSV data file 
df = pd.read_csv("Assessment - Form Responses.csv")
df.drop(columns=["Timestamp", "Email Address"], inplace=True)

# Reference data to calculate the score
dic = {"How spontaneous are you?": "Balanced",
       "Do you enjoy giving or receiving surprises?\n": "Like it",
       "\nHow important is music taste compatibility to you?\n": "Important but not a deal-breaker",
       "How open are you to trying new things (food, travel, experiences)?": "Mostly open",
       "How much personal space do you need in a relationship?": "Quite a bit",
       "How emotionally expressive are you?": "Balanced",
       "How important is having similar long-term goals?": "Important",
       "What’s your preferred mode of communication?": "Face to Face",
       "What’s your ideal time to hang out?": "Late night",
       "What is your ideal weekend plan?": "Outdoor Adventures"}

# Function to calculate the score
def custom_score(row):
    score = 0
    for key, value in dic.items():
        observation = row.get(key)
        #print(observation)
        if observation == value:
            score += 1
    return score

# Applying the function to the data and storing the scores in a column
df['score'] = df.apply(custom_score, axis = 1)

# Based on scores, categorising whether a Match or No match as labels
df['target'] = df['score'].apply(lambda x: "match" if x > 4 else "not match")
df.drop(columns="score", inplace=True)

# Defining input and target columns
X = df.drop('target', axis=1)
y = df['target']

# Ordinal columns
ordinal_columns = [
    'How spontaneous are you?',
    '\nHow important is music taste compatibility to you?\n',
    'How open are you to trying new things (food, travel, experiences)?',
    'How much personal space do you need in a relationship?',
    'How emotionally expressive are you?',
    'How important is having similar long-term goals?',
    'What’s your ideal time to hang out?']

# Nominal columns
nominal_columns = [
    'Do you enjoy giving or receiving surprises?\n',
    'What’s your preferred mode of communication?',
    'What is your ideal weekend plan?']

# Order for ordinal columns
ordinal_orders = [
    ['Very planned', 'Mostly planned', 'Balanced', 'Mostly spontaneous', 'Very spontaneous'],
    ['Doesn’t matter at all', 'Slightly important', 'Neutral / Okay either way', 'Important but not a deal-breaker', 'Very important, must match mine'],
    ['Not open at all', 'Slightly hesitant', 'Sometimes open', 'Mostly open', 'Very adventurous'],
    ['Very little', 'A little', 'Moderate', 'Quite a bit', 'A lot'],
    ['Very reserved', 'Slightly reserved', 'Balanced', 'Mostly expressive', 'Very expressive'],
    ['Not important', 'Slightly important', 'Neutral', 'Important', 'Very important'],
    ['Morning', 'Afternoon', 'Evening', 'Late night']]

# Options for nominal columns
nominal_choices = [["Like it", "Neutral", "Love it", "Hate it", "Dislike it"],
                   ["Face to Face", "Calling", "Texting", "Video calls"],
                   ["Partying","Watching movies", "Outdoor Adventures", "Staying indoors/Chilling"]]


# Pipeline for Ordinal columns
ordinal_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                             ('encoder', OrdinalEncoder(categories=ordinal_orders))])

# Pipeline for nominal columns
nominal_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                             ('encoder', OneHotEncoder(handle_unknown='ignore'))])

# Preprocessing the entire data
preprocessor = ColumnTransformer([('ordinal_Processor', ordinal_pipeline, ordinal_columns),
                                  ('nominal_processor', nominal_pipeline, nominal_columns)])


# Fitting the preprocessor for input data
X_final = preprocessor.fit_transform(X)

# Fitting the Label encoder for output column
le = LabelEncoder()
y_final = le.fit_transform(y)

# Input shape after fitting the preprocessor
input_shape = X_final.shape[1]

# Building ANN model
model = keras.Sequential([keras.layers.Input(shape=(input_shape,)),
    keras.layers.Dense(8, kernel_initializer=keras.initializers.GlorotNormal(seed=42)),
    keras.layers.PReLU(),
    keras.layers.Dense(3, kernel_initializer=keras.initializers.GlorotNormal(seed=42),
                       kernel_regularizer=keras.regularizers.L2()),
    keras.layers.PReLU(),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid", kernel_initializer=keras.initializers.HeNormal(seed=42))])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", Precision(), Recall()])


# Training the model
if st.button("🚀 Train Model"):
    requirements= keras.callbacks.ModelCheckpoint(
        "best_model.keras", monitor="val_loss", save_best_only=True, mode="min"
    )

    with st.spinner("⏳ Training in progress..."):
        history = model.fit(X_final, y_final, batch_size=8, epochs=30,validation_split=0.2,
                            callbacks=[requirements], verbose=0)
    st.success("✅ Model training completed successfully!")

# Collecting the user inputs
st.sidebar.header("Are you a perfect Match? Choose your likes and dislikes to find out!!")
user_input = {}

# Ordinal input columns
for col, order in zip(ordinal_columns, ordinal_orders):
    user_input[col] = st.sidebar.selectbox(col.strip(), order)

# Nominal Input columns
for col, choices in zip(nominal_columns, nominal_choices):
    user_input[col] = st.sidebar.selectbox(col.strip(), choices)

# Prediction
if st.button("💌 Check Our Vibes"):
    input_df = pd.DataFrame([user_input.values()], columns=user_input.keys())
    transformed_input = preprocessor.transform(input_df)
    prediction = model.predict(transformed_input)[0][0]
    if prediction >= 0.5:
        st.success(f"💞 Perfect match detected. Now go get coffee and talk about your dreams! ☕💬")
        st.image("gif1.gif", width = 200)
        st.balloons()
    else:
        st.warning(f"💔 This pair may not vibe, but the right one will. Stay hopeful! 🌸")
        st.image("gif2.gif", width = 200)
