import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import requests


st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
<style>
.sidebar .sidebar-content {
    font-size: 20px;
    background-color: #f8f9fa; /* Light gray background for the sidebar */
    padding-top: 30px; /* Add some top padding to the sidebar content */
    height: 100vh; /* Set the height of the sidebar to fill the entire viewport */
}
.sidebar .sidebar-footer {
    position: absolute;
    bottom: 0;
    width: 100%;
    padding: 10px;
    text-align: center;
    color: #6c757d; /* Gray color for the sidebar footer */
}
</style>
""", unsafe_allow_html=True)
# Load your data
# Assuming 'df' is your DataFrame

# Create a sidebar for navigation

final = pd.read_csv('final.csv')

def movie():
    def fetch_poster(movie_id):
        url = "https://api.themoviedb.org/3/movie/{}?api_key=96427f198fb614e556b2fe677e4562d3&language=en-US".format(movie_id)
        data = requests.get(url)
        data = data.json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path

    def recommend(movie):
        index = movies[movies['title_x'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movie_names = []
        recommended_movie_posters = []
        for i in distances[1:6]:
            # fetch the movie poster
            movie_id = movies.iloc[i[0]].id
            recommended_movie_posters.append(fetch_poster(movie_id))
            recommended_movie_names.append(movies.iloc[i[0]].title_x)

        return recommended_movie_names,recommended_movie_posters

    def recommend_1(movie):
        index = movies[movies['title_x'] == movie].index[0]
        distances = sorted(list(enumerate(similarity_1[index])), reverse=True, key=lambda x: x[1])
        recommended_movie_names = []
        recommended_movie_posters = []
        for i in distances[1:6]:
            # fetch the movie poster
            movie_id = movies.iloc[i[0]].id
            recommended_movie_posters.append(fetch_poster(movie_id))
            recommended_movie_names.append(movies.iloc[i[0]].title_x)

        return recommended_movie_names,recommended_movie_posters

    st.header('Movie Recommender System')
    movies = pickle.load(open('movie_list.pkl','rb'))
    similarity = pickle.load(open('similarity Word Embeding.pkl','rb'))
    similarity_1 = pickle.load(open('similaity Count Vectorizer.pkl','rb'))


    movie_list = movies['title_x'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )
    #col1, col2 = st.columns(2)
    #with col1:
    if st.button('Show Recommendation Using Word Embeding'):
            recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_movie_names[0])
                st.image(recommended_movie_posters[0])
            with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])

            with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
            with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
            with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4])

    #with col2:
    if st.button('Show Recommendation using Count Vectorizer'):
            recommended_movie_names,recommended_movie_posters = recommend_1(selected_movie)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_movie_names[0])
                st.image(recommended_movie_posters[0])
            with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])

            with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
            with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
            with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4])





def uber():
    # Title and introduction
    st.title('UBER Data Visualization DashBoard')
    st.write('This dashboard displays various plots for data exploration.')

    # Add a selectbox for users to choose a specific plot
    selected_plot = st.selectbox('Select a Plot:', ['Animated Bar Plot of Total Amount by Pickup Hour and Month', 'PairPlot (will take some time)', '3D Scatter Plot of Pickup Location and Fare Amount', 
                                                    'Distribution of Tip Amount with KDE and Rug Plot', 'Correlation Heatmap of Numeric Columns','Average Fare Amount by Pickup Hour and Pickup Day',
                                                    'Fare Amount Distribution by Payment Type','Passenger Count by Pickup Hour',
                                                    'ViolinPlot Fare Amount Distribution by Pickup/Dropoff Weekday','ViolinPlot Total Amount Distribution by RatecodeID',
                                                    'Payment Type by Store and Forward Flag','Average Tip Amount by Pickup Hour',
                                                    'Total Amount Distribution by PickupDropoff Month','BoxPlot Total Amount Distribution by RatecodeID'
                                                    'Payment Type Count','Trip Distance vs. Fare Amount','Passenger CountPlot','CountPlot'
                                                    ])

    # Function to display the selected plot based on user choice
    def display_plot(selected_plot):
        if selected_plot == '3D Scatter Plot of Pickup Location and Fare Amount':
            st.subheader('3D Scatter Plot of Pickup Location and Fare Amount')
            # Add code to create and display the bar chart
            fig = px.scatter_3d(final, x='trip_distance', y='fare_amount', z='total_amount', color='passenger_count',
                        hover_name='trip_id', hover_data=['fare_amount', 'trip_distance'],
                        color_continuous_scale='Viridis', size_max=10)
            fig.update_layout(title='3D Scatter Plot of Pickup Location and Fare Amount',
                    scene=dict(xaxis_title='Pickup Longitude', yaxis_title='Pickup Latitude', zaxis_title='Total Amount'))
            st.plotly_chart(fig)
        elif selected_plot == 'Animated Bar Plot of Total Amount by Pickup Hour and Month':
            st.subheader('Animated Bar Plot of Total Amount by Pickup Hour and Month')
            # Add code to create and display the line chart
            fig = px.bar(final, x='pick_hour', y='total_amount', animation_frame='pick_month', animation_group='pickup_longitude',
                range_y=[0, 8000], color='passenger_count', facet_row='store_and_fwd_flag',
                labels={'pick_hour': 'Pickup Hour', 'total_amount': 'Total Amount', 'pick_month': 'Pickup Month'})
            fig.update_layout(title='Animated Bar Plot of Total Amount by Pickup Hour and Month',
                    xaxis=dict(tickmode='array', tickvals=list(range(0, 24))))
            st.plotly_chart(fig)
        elif selected_plot == 'Average Fare Amount by Pickup Hour and Pickup Day':
            st.subheader('Average Fare Amount by Pickup Hour and Pickup Day')
            # Add code to create and display the scatter plot
            pivot_table = final.pivot_table(index='pick_hour', columns='pick_day', values='fare_amount', aggfunc='mean')
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Average Fare Amount'})
            plt.xlabel('Pickup Day')
            plt.ylabel('Pickup Hour')
            plt.title('Average Fare Amount by Pickup Hour and Pickup Day')
            st.pyplot()
        elif selected_plot == 'Distribution of Tip Amount with KDE and Rug Plot':
            st.subheader('Distribution of Tip Amount with KDE and Rug Plot')
            # Add code to create and display the histogram
            plt.figure(figsize=(10, 6))
            sns.distplot(final['tip_amount'], bins=30, kde=True, rug=True, hist_kws={'edgecolor': 'black'}, kde_kws={'color': 'red'})
            plt.xlabel('Tip Amount')
            plt.ylabel('Density')
            plt.title('Distribution of Tip Amount with KDE and Rug Plot')
            st.pyplot()
        elif selected_plot == 'Fare Amount Distribution by Payment Type':
            st.subheader('Fare Amount Distribution by Payment Type')
            # Add code to create and display the pie chart
            sample_df = final.sample(frac=0.01, random_state=42)

            plt.figure(figsize=(12, 6))
            sns.swarmplot(x='payment_type_name', y='fare_amount', data=sample_df, palette='Set2', dodge=True, size=3)
            plt.xlabel('Payment Type')
            plt.ylabel('Fare Amount')
            plt.title('Fare Amount Distribution by Payment Type')
            st.pyplot()
        elif selected_plot == 'Passenger Count by Pickup Hour':
            st.subheader('Passenger Count by Pickup Hour')
            # Add code to create and display the box plot
            plt.figure(figsize=(12, 6))
            sns.countplot(x='pick_hour', hue='passenger_count', data=final, palette='viridis', saturation=0.7)
            plt.xlabel('Pickup Hour')
            plt.ylabel('Count')
            plt.title('Passenger Count by Pickup Hour')
            plt.legend(title='Passenger Count')
            st.pyplot()
        elif selected_plot == 'Fare Amount Distribution by Pickup/Dropoff Weekday':
            st.subheader('Fare Amount Distribution by Pickup/Dropoff Weekday')
            # Add code to create and display the box plot
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.violinplot(x='pick_weekday', y='fare_amount', data=final)
            plt.xlabel('Pickup Weekday')
            plt.ylabel('Fare Amount')
            plt.title('Fare Amount Distribution by Pickup Weekday')

            plt.subplot(1, 2, 2)
            sns.violinplot(x='drop_weekday', y='fare_amount', data=final)
            plt.xlabel('Dropoff Weekday')
            plt.ylabel('Fare Amount')
            plt.title('Fare Amount Distribution by Dropoff Weekday')
            st.pyplot()
        elif selected_plot == 'Total Amount Distribution by RatecodeID':
            st.subheader('Total Amount Distribution by RatecodeID')
            # Add code to create and display the box plot
            plt.figure(figsize=(10, 6))
            sns.violinplot(x='RatecodeID', y='total_amount', data=final)
            plt.xlabel('RatecodeID')
            plt.ylabel('Total Amount')
            plt.title('Total Amount Distribution by RatecodeID')
            st.pyplot()
        elif selected_plot == 'Payment Type by Store and Forward Flag':
            st.subheader('Payment Type by Store and Forward Flag')
            # Add code to create and display the box plot
            plt.figure(figsize=(12, 6))
            sns.countplot(x='store_and_fwd_flag', hue='payment_type_name', data=final)
            plt.xlabel('Store and Forward Flag')
            plt.ylabel('Count')
            plt.title('Payment Type by Store and Forward Flag')
            st.pyplot()
        elif selected_plot == 'Correlation Heatmap of Numeric Columns':
            st.subheader('Correlation Heatmap of Numeric Columns')
            # Add code to create and display the box plot
            numeric_columns = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
                    'total_amount', 'passenger_count', 'trip_distance', 'pickup_latitude', 'pickup_longitude',
                    'dropoff_latitude', 'dropoff_longitude']
            correlation_matrix = final[numeric_columns].corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Heatmap of Numeric Columns')
            st.pyplot()
        elif selected_plot == 'PairPlot (will take some time)':
            st.subheader('PairPlot')
            # Add code to create and display the box plot
            numeric_columns = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
                    'total_amount', 'passenger_count', 'trip_distance', 'pickup_latitude', 'pickup_longitude',
                    'dropoff_latitude', 'dropoff_longitude']
            sample_df = final[numeric_columns]
            sample_df = sample_df.sample(frac=0.1, random_state=42)
            sns.pairplot(sample_df)
            st.pyplot()
        elif selected_plot == 'Average Tip Amount by Pickup Hour':
            st.subheader('Average Tip Amount by Pickup Hour')
            # Add code to create and display the box plot
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='pick_hour', y='tip_amount', data=final.groupby('pick_hour')['tip_amount'].mean().reset_index())
            plt.xlabel('Pickup Hour')
            plt.ylabel('Average Tip Amount')
            plt.title('Average Tip Amount by Pickup Hour')
            st.pyplot()
        elif selected_plot == 'Total Amount Distribution by Pickup/Dropoff Month':
            st.subheader('Total Amount Distribution by PickupDropoff Month')
            # Add code to create and display the box plot
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.boxplot(x='pick_month', y='total_amount', data=final)
            plt.xlabel('Pickup Month')
            plt.ylabel('Total Amount')
            plt.title('Total Amount Distribution by Pickup Month')

            plt.subplot(1, 2, 2)
            sns.boxplot(x='drop_month', y='total_amount', data=final)
            plt.xlabel('Dropoff Month')
            plt.ylabel('Total Amount')
            plt.title('Total Amount Distribution by Dropoff Month')
            st.pyplot()
        elif selected_plot == 'Total Amount Distribution by RatecodeID':
            st.subheader('Total Amount Distribution by RatecodeID')
            # Add code to create and display the box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='rate_code_name', y='total_amount', data=final)
            plt.xlabel('RatecodeName')
            plt.xticks(rotation=45)
            plt.ylabel('Total Amount')
            plt.title('Total Amount Distribution by RatecodeID')
            st.pyplot()
        elif selected_plot == 'Payment Type Count':
            st.subheader('Payment Type Count')
            # Add code to create and display the box plot
            pplt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.countplot(x='rate_code_name', data=final)
            plt.xticks(rotation=45)
            plt.title('RatecodeID Count')

            plt.subplot(1, 2, 2)
            sns.countplot(x='payment_type_name', data=final)
            plt.xticks(rotation=45)
            plt.title('Payment Type Count')
            st.pyplot()
        elif selected_plot == 'Trip Distance vs. Fare Amount':
            st.subheader('Trip Distance vs. Fare Amount')
            # Add code to create and display the box plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='trip_distance', y='fare_amount', data=final)
            plt.xlabel('Trip Distance')
            plt.ylabel('Fare Amount')
            plt.title('Trip Distance vs. Fare Amount')
            st.pyplot()
        elif selected_plot == 'CountPlot':
            st.subheader('CountPlot')
            # Add code to create and display the box plot
            ax = sns.countplot(x = 'payment_type_name',data = final)

            for bars in ax.containers:
                ax.bar_label(bars)
            st.pyplot()
        elif selected_plot == 'Passenger CountPlot':
            st.subheader('Passenger CountPlot')
            # Add code to create and display the box plot
            pa_c = sns.countplot(x = 'passenger_count',data = final)

            for bars in pa_c.containers:
                pa_c.bar_label(bars)
            st.pyplot()

    # Call the function to display the selected plot
    display_plot(selected_plot)


st.sidebar.title("Select Model")
page = st.sidebar.radio("Go to:", ["Movie Recommender System", "UBER Data Visualization DashBoard"])

# Show the selected page content
if page == "Movie Recommender System":
    movie()
elif page == "UBER Data Visualization DashBoard":
    uber()

st.sidebar.markdown("""
<div class="sidebar-footer">
    <p>© 2023 H.T. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)