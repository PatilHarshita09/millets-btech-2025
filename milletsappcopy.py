import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
#----added for charts----
import plotly.express as px  
#----added for geographic maps----
from streamlit_echarts import st_echarts
import folium
from streamlit_folium import folium_static
#----added for animation----
import json
import requests
from streamlit_lottie import st_lottie
import sys
#----added for chatbot----
from openai import OpenAI
import streamlit.components.v1 as components



#----we are initializing openai client instance using api key from sys here----
client = OpenAI(api_key=st.secrets["API_KEY"])
print(sys.path)

#----this is the function for loading animation----
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
lottie_coding = load_lottiefile("homepage.json")
lottie_about = load_lottiefile("about.json")
lottie_error = load_lottiefile("error.json")
lottie_stats = load_lottiefile("stats.json")



#---- dictionary of millet regions for map----
millet_regions = {
    'Foxtail Millet': ['karnataka', 'meghalaya'],
    'Finger Millet': ['meghalaya', 'uttarakhand', 'madhyapradesh', 'karnataka', 'kerala', 'tamilnadu', 'gujrat', 'odisha', 'west bengal', 'maharashtra'],
    'Banyard Millet': ['uttrakhand', 'tamil nadu'],
    'Browntop Millet': ['karnataka'],
    'Little Millet': ['maharashtra', 'karnataka', 'tamil nadu', 'odisha', 'madhya pradesh', 'andrapradesh'],
    'Kodo Millet': ['tamil nadu', 'karnataka', 'odisha', 'madhya pradesh'],
    'pearl Millet': ['jammu & kashmir', 'haryana', 'rajasthan', 'gujarat', 'maharashtra', 'karnataka', 'tamil nadu', 'telangana', 'uttar pradesh', 'gujarat', 'madhyapradesh'],
    'Proso Millet': ['tamil nadu', 'karnataka', 'andra pradesh', 'uttarakhand'],
    'Sorghum Millet': ['kerala', 'telangana', 'karnataka', 'maharashtra', 'rajasthan', 'haryana', 'uttarpradesh', 'madhya pradesh', 'andrapradesh'],
    'Buckwheat Millet':['jammu & kashmir','uttrakhand','chattisgarh'],
    'Kokan Red Rice':['maharashtra'],
    'Amaranth Millet':['kerala','karnataka','tamil nadu','maharashtra'],
}

# ----Function to display the map----(we used folium to display map centered around india and places markers where the selected millets type is grown)
def display_map(selected_millet):
    # we are Initializing map centered around India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    # markers for regions where the selected millet is grown
    if selected_millet in millet_regions:
        regions = millet_regions[selected_millet]
        for region in regions:
            #coordinates for the region
            region_coordinates = {
                'karnataka': [15.3173, 75.7139],
                'meghalaya': [25.4670, 91.3662],
                'uttarakhand': [30.0668, 79.0193],
                'madhyapradesh': [22.9734, 78.6569],
                'kerala': [10.8505, 76.2711],
                'tamilnadu': [11.1271, 78.6569],
                'gujrat': [22.2587, 71.1924],
                'odisha': [20.9517, 85.0985],
                'west bengal': [22.9868, 87.8550],
                'maharashtra': [19.7515, 75.7139],
                'uttrakhand': [30.0668, 79.0193],
                'andrapradesh': [15.9129, 79.7400],
                'haryana': [29.0588, 76.0856],
                'rajasthan': [27.0238, 74.2179],
                'telangana': [18.1124, 79.0193],
                'uttar pradesh': [26.8467, 80.9462],
                'jammu & kashmir': [33.7782, 76.5762],
                'chattisgarh': [21.2787, 81.8661],
            }

            if region in region_coordinates:
                folium.Marker(location=region_coordinates[region], popup=region).add_to(m)

    # Displaying the map of about page
    folium_static(m)

#----function for Tensorflow Model Prediction----

def model_prediction(test_image):  
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)    #converting image to array
    input_arr = np.array([input_arr])                               #converting single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)    

#function for Load CSV data and getting df 
def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path, delimiter=',')
    return df  

#------------------------------------------------------------------sidebar------------------------------------------------------------------------
st.sidebar.title("Dashboard")
#dropdown menu to select different pages
app_mode= st.sidebar.selectbox("Select page",["Home","About","MyMilletGuide","Analysis","Prediction","Recipes"])


#-------------------------------------------------------------------Analysis Page=-----------------------------------------------------------------
if app_mode == "Analysis":
    st.markdown('<h1 class="dashboard-title">Millet Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    st_lottie(
        lottie_stats,
        speed=1.5,
        reverse=False,
        loop=True,
        quality="low",
        height=None,
        width=None,
        key=None,
    )
    
    # custom css for styling
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f4f8; /* Light background color  */
        }
        .dashboard-title {
            color: #0000ff;  /* Dark Green for title */
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        .button {
            background-color: e6e6fa; /* Green background */
            color: white; /* White text */
            border: none; 
            border-radius: 5px;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 18px;
            margin: 20px auto;
            cursor: pointer;
            transition-duration: 0.4s;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Button shadow */
        }
        .button:hover {
            background-color: white; /* White background on hover */
            color: black; /* Black text on hover */
            border: 2px solid #4CAF50; 
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    

    # Button to open Power BI report
    report_url = "https://app.powerbi.com/groups/me/reports/f04dac31-b65f-4306-8c5d-9e59292bf9c5/e0427c3ea314804031d9?experience=power-bi"
    
    st.markdown(f"""
        <a href="{report_url}" target="_blank" class="button">Open Dashboard</a>
    """, unsafe_allow_html=True)

    # Dashboard information
    with st.expander("Dashboard Information"):
        st.write(""" 
            This dashboard provides detailed analysis of millet iinformation, 
            consumption patterns, and nutritional comparisons based on the CONDUCTED SURVEY.
            Note: You need to be logged into Power BI to view this dashboard.
        """)


#------------------------------------------------------------------home page------------------------------------------------------------------
#Home page which displays a welcome screen and introduces users about the aim of this webapp
if(app_mode=="Home"):

    st_lottie(
    lottie_coding,

    speed=1.5,
    reverse=False,
    loop=True,
    quality="low",
    # renderer="svg",
    height=None,
    width=None,
    key=None,
    )

    st.header("MILLETS RECOGNITION SYSTEM")
    container = st.container()
    container.write("In 2023, the world celebrated Millets Year, and now we're introducing our new image recognition system which aims to raise awareness about the importance of incorporating millets into our diets. By using machine learning, it helps identify different types of millets and provides essential nutritional information about millets, empowering people to make informed dietary choices. ")
    image_path = "images\millet.jpg" 
    st.image(image_path)     

#------------------------------------------------------------------About Page------------------------------------------------------------------
#
if(app_mode=="About"):
    selected_millet = st.sidebar.selectbox("Select Millet Type", list(millet_regions.keys()))
    # st.header("About")
    st_lottie(
                            lottie_about,

                            speed=1.5,
                            reverse=False,
                            loop=True,
                            quality="low",
                            # renderer="svg",
                            height=None,
                            width=None,
                            key=None,
    )
    st.write("This is an application to predict,analyze and explore various millet types.")
    st.write("Here's the distribution of millet crops in India:")
    display_map(selected_millet)



# -----------------------------------------------------------------mymilletguide Page-----------------------------------------------------------------
if app_mode == "MyMilletGuide":
    st.title("🌾🌾Millet Chatbot🌾🌾")  

    # Initializing session state for messages and context
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "Hello how can I assist you?"}
        ]
    if "current_millet" not in st.session_state:
        st.session_state["current_millet"] = None #current millet for follow-up

    # Displaying the chatbot starter message
    if not st.session_state.get("started"):
        st.session_state.messages.append(
            {"role": "assistant", "content": "Want to know about millet?"}  
        )
        st.session_state["started"] = True

    # clear button 
    if st.button("Clear Chat"): 
        st.session_state["messages"] = [
            {"role": "system", "content": "Hello, How can I help you?"}
        ]
        st.session_state["current_millet"] = None
        

  
    with st.container():
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'><strong>User:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)

    # greetings handling
    def is_greeting(prompt):
        greetings = ["hi", "hello", "hey", "bye", "goodbye", "greetings","Thankyou"]
        return any(greeting in prompt.lower() for greeting in greetings)

    # classifying user query
    def classify_query(prompt):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Classify the query as 'millet' if it is about a specific millet, "
                    "'follow_up' if it refers to the last mentioned millet, "
                    "or 'out_of_context' if it is unrelated."
                )},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip().lower()

    # handling chat input here
    if prompt := st.chat_input("Ask your question about millets:"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # displaying user msg
        with st.chat_message("user"):
            st.markdown(f"<div class='user-message'><strong>User:</strong> {prompt}</div>", unsafe_allow_html=True)

        # Handle greetings
        if is_greeting(prompt):
            greeting_response = "Hello! How can I assist you with millets today?"
            with st.chat_message("assistant"):
                st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong> {greeting_response}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": greeting_response})

        else:
         
            classification = classify_query(prompt)

            if classification == "millet":
                # Storing current millet topic for follow-ups
                st.session_state["current_millet"] = prompt

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=st.session_state.messages,
                        stream=True,
                    )

            
                    for chunk in response:
                        delta_content = getattr(chunk.choices[0].delta, "content", "")
                        if delta_content:
                            full_response += delta_content
                            message_placeholder.markdown(f"<div class='assistant-message'><strong>Assistant:</strong> {full_response}</div>", unsafe_allow_html=True)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})

            elif classification == "follow_up" and st.session_state["current_millet"]:
  
                follow_up_response = (
                    f"It seems you're asking more about {st.session_state['current_millet']}. "
                    "Here's some additional information: "
                    "This millet is highly nutritious and can be used in various dishes. "
                    "Would you like to know more about its benefits or recipes?"
                )
                with st.chat_message("assistant"):
                    st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong> {follow_up_response}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": follow_up_response})

            else:
                # Out-of-context response handling
                out_of_context_message = (
                    "I'm here to assist only with questions related to millets. "
                    "Please try to keep the conversation focused on that topic."
                )
                with st.chat_message("assistant"):
                    st.markdown(f"<div class='assistant-message'><strong>Assistant: </strong>{out_of_context_message}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": out_of_context_message})

#styling
st.markdown(
    """
    <style>
    .user-message {
        background-color: #FFEDF9;  /* Light cyan for user messages */
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        max-width: 70%;
        align-self: flex-start;
    }
    .assistant-message {
        background-color: #ffe0b2;  /* Light orange for assistant messages */
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        max-width: 70%;
        align-self: flex-end;
    }
    .stButton>button {
        background-color: #b9fbb9;
        color: #000;
        border: none;
        border-radius: 5px;
        padding: 10px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #ffab91;
    }
    </style>
    """,
    unsafe_allow_html=True
)
#-----------------------------------------------------------------prediction page-----------------------------------------------------------------
if app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    
    # Show button
    if st.button("Show Image"):
        if test_image is not None:
            if not test_image.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                st.error("Please upload an image with JPG, JPEG, PNG, or WEBP format.")
                st_lottie(
                            lottie_error,

                            speed=1.5,
                            reverse=False,
                            loop=True,
                            quality="low",
                            # renderer="svg",
                            height=200,
                            width=200,
                            key=None,
                )
            else:
                st.image(test_image, width=400)
        else:
            st.error("Please upload an image first.")
            st_lottie(
                            lottie_error,

                            speed=1.5,
                            reverse=False,
                            loop=True,
                            quality="low",
                            # renderer="svg",
                            height=200,
                            width=200,
                            key=None,
                )

    # Predict button
    if st.button("Predict"):
        if test_image is not None:
            # Checking file extension
            if not test_image.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                st.error("Please upload an image with JPG, JPEG, PNG, or WEBP format.")
                st_lottie(
                            lottie_error,

                            speed=1.5,
                            reverse=False,
                            loop=True,
                            quality="low",
                            # renderer="svg",
                            height=200,
                            width=200,
                            key=None,
                )
            else:
                st.write("Our Prediction")
                result_index = model_prediction(test_image)
                
                # Reading Labels
                with open("labels.txt") as f:
                    content = f.readlines()
                label = [i.strip() for i in content]
                predicted_millet = label[result_index]
                print(predicted_millet)         # printing the predicted millet
                # Loading millet information from CSV file
                df = load_data("millets_information.csv")


                predicted_millet_info = df[df['Millet_Name'] == predicted_millet].squeeze()
                st.success("The above image is of {}".format(predicted_millet_info['Name']))

                            

                # Displaying millet information
                st.subheader("Millet Information")
                st.write("**Name:**", predicted_millet_info['Name'])
                st.write("**Introduction:**", predicted_millet_info['Introduction'])
                st.write("**Botanical Name:**", predicted_millet_info['Botanical Name'])
                st.write("**Common Names:**", predicted_millet_info['Common Names'])
                st.write("**Cultivation Areas:**)", predicted_millet_info['Cultivation Areas'])
                st.write("**Appearance:**", predicted_millet_info['Appearance'])

                # Displaying benefits in a list format
                st.subheader("Benefits")
                benefits = predicted_millet_info['Benefits'].split('*')
                for benefit in benefits:
                    st.write(f"- {benefit.strip()}")



                # Displaying nutritional values as a doughnut chart
                st.subheader("Nutritional Value:")
                st.write("Per 100g of", predicted_millet_info['Name'],":")
                nutrients = ['Energy(kcal)', 'Carbohydrates(g)', 'Protein(g)', 'Fat(g)',  'Fiber(g)'] 
                nutrient_values = [float(predicted_millet_info[nutrient]) for nutrient in nutrients]  # Convert to int

                nutritional_options = {
                    "tooltip": {"trigger": "item"},
                    "legend": {"top": "5%", "left": "center"},
                    "series": [
                        {
                            "name": "Nutritional Values",
                            "type": "pie",
                            "radius": ["40%", "70%"],
                            "color": ["#023047", "#219EBC", "#8ECAE6","#FFB703","#FB8500"],

                            "avoidLabelOverlap": False,
                            "itemStyle": {
                                "borderRadius": 10,
                                "borderColor": "#fff",
                                "borderWidth": 2,
                            },
                            "label": {"show": False, "position": "center"},
                            "emphasis": {
                                "label": {"show": True, "fontSize": "16", "fontWeight": "bold"}
                            },
                            "labelLine": {"show": False},
                            "data": [
                                {"value": value, "name": nutrient}
                                for nutrient, value in zip(nutrients, nutrient_values)
                            ],
                        }
                    ],
                }
                st_echarts(options=nutritional_options, height="500px")

                st.subheader("Mineral Values:")
                minerals = ['Calcium(mg)', 'Iron(mg)', 'Pottasium(mg)', 'Magnesium(mg)', 'Zinc(mg)'] 
                mineral_values = [float(predicted_millet_info[mineral]) for mineral in minerals] 
                mineral_options = {
                    "tooltip": {"trigger": "item"},
                    "legend": {"top": "5%", "left": "center"},
                    "series": [
                        {
                            "name": "Mineral Values",
                            "type": "pie",
                            "radius": ["40%", "70%"],
                            "color": ["#DAF7A6", "#FFC300", "#FF5733","#C70039","#900C3F"],
                            "avoidLabelOverlap": False,
                            "itemStyle": {
                                "borderRadius": 10,
                                "borderColor": "#fff",
                                "borderWidth": 2,
                            },
                            "label": {"show": False, "position": "center"},
                            "emphasis": {
                                "label": {"show": True, "fontSize": "16", "fontWeight": "bold"}
                            },
                            "labelLine": {"show": False},
                            "data": [
                                {"value": value, "name": mineral}
                                for mineral, value in zip(minerals, mineral_values)
                            ],
                        }
                    ],
                }
                st_echarts(options=mineral_options, height="500px")

                recipes_data = load_data("millets_recipe.csv")
                predicted_millet_recipe = recipes_data[recipes_data['Millet_Name'] == predicted_millet_info['Millet_Name']].squeeze()
                print(predicted_millet_recipe)

                st.subheader("Millet Recipe:")
                st.write("**Recipe Name:**", predicted_millet_recipe['Recipes_name'])
                st.image(predicted_millet_recipe['Image'], caption=predicted_millet_recipe['Recipes_name'], width=400)
                st.write("**Ingredients :**", predicted_millet_recipe['Ingredients'])
                st.write("**Recipe:**")
                recipes = predicted_millet_recipe['Recipe'].split('*')
                for recipe in recipes:
                    st.write(f"- {recipe.strip()}")


                    
                
        else:
            st.error("Please upload an image before making a prediction.")
            st_lottie(
                            lottie_error,

                            speed=1.5,
                            reverse=False,
                            loop=True,
                            quality="low",
                            height=200,
                            width=200,
                            key=None,
                )
# -----------------------------------------------------------------Recipes Page-----------------------------------------------------------------
elif app_mode == "Recipes":
    st.header("Millet Recipes")
    
    
    recipes_data = load_data("millets_recipe.csv")
    

    selected_millet = st.selectbox("Select Millet Type", recipes_data['Name'].unique())
    

    selected_recipes = recipes_data[recipes_data['Name'] == selected_millet]
    
    if not selected_recipes.empty:
        # recipes as flex cards
        for index, row in selected_recipes.iterrows():
            st.write(f"### {row['Recipes_name']}")
            st.image(row['Image'], caption=row['Recipes_name'], width=400)
            st.write("**Ingredients:**", row['Ingredients'])
            st.write("**Recipe:**")
            recipe_sentences = row['Recipe'].split('*')
            for sentence in recipe_sentences:
                st.write(f"- {sentence.strip()}")
            st.markdown("---")
    else:
        st.write("No recipes found for the selected millet.")