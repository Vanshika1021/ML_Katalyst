import numpy as np
import pandas as pd
import random
import time
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Simulated Traffic Sensors at intersections (IoT Sensors)
class TrafficSensor:
    def __init__(self, intersection_name):
        self.name = intersection_name
        self.traffic_flow = 0

    def update_traffic_flow(self):
        """Simulate random traffic flow between 0-1000 vehicles"""
        self.traffic_flow = random.randint(100, 1000)

    def get_traffic_flow(self):
        return self.traffic_flow

# AI Model for predicting peak congestion times
class TrafficAIModel:
    def __init__(self):
        self.model = LinearRegression()

    def train_model(self, historical_data):
        """Train AI model to predict traffic congestion based on historical data"""
        X = np.array(historical_data['time']).reshape(-1, 1)  # Time as feature
        y = np.array(historical_data['traffic_flow'])  # Traffic flow as target
        self.model.fit(X, y)

    def predict_peak_congestion(self, time):
        """Predict the congestion level at a specific time"""
        return self.model.predict(np.array([[time]]))[0]

# Dynamic Traffic Light System
class DynamicTrafficLight:
    def __init__(self, intersection_name):
        self.name = intersection_name
        self.green_light_duration = 0  # Duration of the green light in seconds
        self.red_light_duration = 0    # Duration of the red light in seconds

    def adjust_signal(self, traffic_flow):
        """Adjust the signal duration based on traffic flow"""
        if traffic_flow > 800:
            self.green_light_duration = 60  # Longer green light for heavy traffic
            self.red_light_duration = 30
        elif 400 <= traffic_flow <= 800:
            self.green_light_duration = 45
            self.red_light_duration = 45
        else:
            self.green_light_duration = 30  # Shorter green light for light traffic
            self.red_light_duration = 60

    def show_traffic_signal(self):
        """Show current traffic light status"""
        print(f"{self.name} Traffic Light: Green for {self.green_light_duration}s, Red for {self.red_light_duration}s")

# Simulating the Traffic Management System
def simulate_traffic_system():
    # Create traffic sensors for 3 intersections
    intersections = ['Intersection 1', 'Intersection 2', 'Intersection 3']
    sensors = [TrafficSensor(intersection) for intersection in intersections]
    
    # Create AI model and train with historical data (for simulation, using made-up data)
    historical_data = pd.DataFrame({
        'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'traffic_flow': [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]  # Traffic flow at different times
    })
    
    ai_model = TrafficAIModel()
    ai_model.train_model(historical_data)
    
    # Create dynamic traffic lights for each intersection
    traffic_lights = [DynamicTrafficLight(intersection) for intersection in intersections]

    # Simulate the system and update traffic flow and signals
    for time_step in range(1, 11):
        print(f"\nSimulating for Time Step: {time_step} hours")

        # Update traffic flow for each sensor (simulating real-time data collection)
        for sensor in sensors:
            sensor.update_traffic_flow()
            print(f"{sensor.name}: Traffic Flow = {sensor.get_traffic_flow()} vehicles")

        # Predict peak congestion and adjust traffic lights
        for light, sensor in zip(traffic_lights, sensors):
            peak_congestion = ai_model.predict_peak_congestion(time_step)
            print(f"AI Prediction for {sensor.name} peak congestion: {peak_congestion:.2f} vehicles")

            # Adjust the traffic light based on real-time traffic flow
            light.adjust_signal(sensor.get_traffic_flow())
            light.show_traffic_signal()

        # Wait before simulating the next time step
        time.sleep(1)  # In a real-world system, this would be real-time, but here we simulate it.

# Running the simulation
if __name__ == "__main__":
    simulate_traffic_system()
