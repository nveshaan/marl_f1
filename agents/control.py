import numpy as np
from scripts.optimal import compute_optimal_line

class PIDAgent:
    def __init__(self, env_track_data, target_speed=45.0):
        raw_xy = np.array([(p[2], p[3]) for p in env_track_data])
        self.path = compute_optimal_line(raw_xy)
        
        # PID Constants
        self.target_speed = target_speed
        self.steer_kp = 1.0  # Muscle
        self.steer_kd = 1.5  # Damping (prevents wobbles)
        self.speed_kp = 0.5  
        
        # Memory Variables for PID
        self.prev_steer_error = 0
        
        # --- NEW: Action Memory for Smoothing ---
        self.prev_steer = 0.0
        self.prev_gas = 0.0
        self.prev_brake = 0.0
        self.smoothing = 0.3 # 0.1 is very heavy smoothing, 0.9 is twitchy

    def step(self, x, y, theta, current_speed):
        car_pos = np.array([x, y])
        
        # 1. Path Finding
        distances = np.linalg.norm(self.path - car_pos, axis=1)
        nearest_idx = np.argmin(distances)
        target_point = self.path[(nearest_idx + 5) % len(self.path)]

        # 2. Lateral Control (Math Alignment Fix)
        target_vec = target_point - car_pos
        target_angle = np.arctan2(target_vec[0], target_vec[1])
        heading_error = target_angle + theta 
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # 3. Cross-Track Error (Magnet Pull)
        path_vec = self.path[(nearest_idx + 1) % len(self.path)] - self.path[nearest_idx]
        car_to_path_vec = car_pos - self.path[nearest_idx]
        side = np.sign(np.cross(path_vec, car_to_path_vec))
        cte = side * distances[nearest_idx]
        total_error = heading_error - (0.2 * cte / (current_speed + 1))

        # 4. Compute PID Outputs
        steer_diff = total_error - self.prev_steer_error
        raw_steer_action = (self.steer_kp * total_error) + (self.steer_kd * steer_diff)
        self.prev_steer_error = total_error
        
        speed_error = self.target_speed - current_speed
        raw_speed_output = np.tanh(self.speed_kp * speed_error)

        # 5. Coordination & Traction Circle
        steer_intensity = abs(np.tanh(raw_steer_action))
        grip_multiplier = 1.0 - (steer_intensity * 0.7) # Save grip for turns
        
        raw_steer = np.tanh(raw_steer_action)
        raw_gas = max(0, raw_speed_output) * grip_multiplier
        raw_brake = abs(min(0, raw_speed_output)) * 0.5
        
        # 6. Apply Memory Smoothing (Temporal Consistency)
        # New = (Alpha * Current) + ((1 - Alpha) * Previous)
        steer = (self.smoothing * raw_steer) + ((1 - self.smoothing) * self.prev_steer)
        gas = (self.smoothing * raw_gas) + ((1 - self.smoothing) * self.prev_gas)
        brake = (self.smoothing * raw_brake) + ((1 - self.smoothing) * self.prev_brake)

        # Update Memory
        self.prev_steer, self.prev_gas, self.prev_brake = steer, gas, brake

        return [steer, gas, brake]