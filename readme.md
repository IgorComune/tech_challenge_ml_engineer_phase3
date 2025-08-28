# Tech Challenge - Phase 3

# Objective
* ETA Prediction = satisfação do cliente.
* Produtividade = reduzir custo de operação.
* Anomalias = evitar perdas escondidas.
* Áreas problemáticas = planejamento estratégico da malha logística.

## Dataset
This Amazon Delivery Dataset provides a comprehensive view of the company's last-mile logistics operations. It includes data on over 43,632 deliveries across multiple cities, with detailed information on order details, delivery agents, weather and traffic conditions, and delivery performance metrics. The dataset enables researchers and analysts to uncover insights into factors influencing delivery efficiency, identify areas for optimization, and explore the impact of various variables on the overall customer experience.

### Link: https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset

#### Columns
##### Order_ID: Unique identifier for each order
##### Agent_Age: Age of the delivery agent
##### Agent_Rating: Rating or performance score of the delivery agent
##### Store_Latitude: Geographic coordinates of the store where the order was placed
##### Store_Longitude: Geographic coordinates of the store where the order was placed
##### Drop_Latitude: Geographic coordinates of the delivery location
##### Drop_Longitude: Geographic coordinates of the delivery location
##### Order_Date: Date when the order was placed
##### Order_Time: Time when the order was placed
##### Pickup_Time: Time when the order was picked up for delivery
##### Weather: Weather conditions during the delivery (e.g., sunny, rainy, snowy)
##### Traffic: Traffic conditions during the delivery (e.g., low, medium, jam)
##### Vehicle: Type of vehicle used for the delivery (e.g., van, motorcycle, bicycle, scooter)
##### Area: Area where the delivery took place (Urban, Metropolitan,etc)
##### Delivery_Time: Time taken to complete the delivery
##### Category: Product category of the ordered item (e.g., electronics, apparel, groceries)

# Install 
python -m venv .venv
.venv/Scripts/activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt