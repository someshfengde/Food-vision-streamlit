import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

class_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets',
               'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli',
               'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry',
               'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
               'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts',
               'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips',
               'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
               'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
               'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus',
               'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons',
               'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes',
               'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
               'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
               'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara',
               'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu',
               'tuna_tartare', 'waffles']

# st.cache();


def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


def preprocess_buffer(buffer):
    image = Image.open(buffer).convert('RGB')
    image_array = np.array(image)
    resized_image = tf.image.resize(image_array, [224, 224])
    return resized_image,image


def predict_and_view(buffer, model_path):
    model = load_model(model_path)
    resized_image,buffered_image = preprocess_buffer(buffer)
    expanded_image = tf.expand_dims(resized_image, axis=0)
    prediction = model.predict(expanded_image)
    prediction_name = class_names[prediction.argmax(axis=1)[0]]
    prediction_accuracy = prediction[0][prediction.argmax(axis=1)[0]]
    return buffered_image, prediction_name, prediction_accuracy
