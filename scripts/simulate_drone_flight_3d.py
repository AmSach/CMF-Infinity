import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import random

# Force UTF-8 stdout encoding for Windows compatibility
sys.stdout.reconfigure(encoding='utf-8')

from cmf import CMFConfig, DeliberativeContinuousMeaningField
from cmf.data import ByteTokenizer
from cmf.model import apply_rotary_pos_emb

# Valid 3D command set
VALID_COMMANDS_3D = ["FLY_FORWARD", "FLY_LEFT", "FLY_RIGHT", "FLY_UP", "FLY_DOWN", "RETURN_TO_BASE", "LAND"]

def generate_flight_data_3d(num_samples=500):
    """Generates synthetic 3D drone state-action pairs representing 3D flight rules."""
    data = []
    for _ in range(num_samples):
        # Coordinates in 3D
        x, y, z = random.uniform(0, 10), random.uniform(0, 10), random.uniform(2, 20)
        vx, vy, vz = random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-1, 1)
        wx, wy, wz = random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-1, 1)
        bat = random.randint(2, 100)
        
        # Target coordinate
        tx, ty, tz = 8.0, 8.0, 12.0
        dist_to_target = math.sqrt((tx-x)**2 + (ty-y)**2 + (tz-z)**2)
        
        # Nearest obstacle center
        ox, oy, oz = 4.0, 4.0, 10.0
        rad = 2.0
        
        # Distance to obstacle center
        doc = math.sqrt((ox-x)**2 + (oy-y)**2 + (oz-z)**2)
        obs_dist = max(0.0, doc - rad)
        
        # Obstacle vector relative to drone
        odx, ody, odz = ox - x, oy - y, oz - z
        
        # Determine optimal action
        if bat < 12:
            if z > 2.0:
                action = "FLY_DOWN"
            else:
                action = "LAND"
        elif bat < 22:
            action = "RETURN_TO_BASE"
        elif obs_dist <= 2.5:
            # Obstacle avoidance: steer up if obstacle is below target, or steer sideways
            if odz < 0.5:
                action = "FLY_UP"
            else:
                action = "FLY_RIGHT"
        else:
            # Navigate towards target
            if z < tz - 1.5:
                action = "FLY_UP"
            elif z > tz + 1.5:
                action = "FLY_DOWN"
            else:
                # Sideways alignment or forward
                action = "FLY_FORWARD"
                
        prompt = f"POS: {x:.1f},{y:.1f},{z:.1f} | VEL: {vx:.1f},{vy:.1f},{vz:.1f} | WIND: {wx:.1f},{wy:.1f},{wz:.1f} | OBS: {odx:.1f},{ody:.1f},{odz:.1f},{obs_dist:.1f} | BAT: {bat}% -> CMD:"
        data.append({"prompt": prompt, "action": action})
    return data

class DroneSimulator3D:
    """A 3D coordinate flight simulator supporting obstacles and wind vectors."""
    def __init__(self, target_x=6.0, target_y=6.0, target_z=12.0, obstacles=None):
        self.x = 0.0
        self.y = 0.0
        self.z = 10.0 # Altitude start
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.battery = 100
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z
        # Obstacles list: list of tuples (cx, cy, cz, radius)
        self.obstacles = obstacles if obstacles else [(3.0, 3.0, 10.0, 2.0)]
        # Wind vectors (wx, wy, wz)
        self.wx = random.uniform(-1.5, 1.5)
        self.wy = random.uniform(-1.5, 1.5)
        self.wz = random.uniform(-0.5, 0.5)
        self.steps = 0
        self.status = "FLYING" # FLYING, REACHED_TARGET, CRASHED, OUT_OF_BATTERY, LANDED_SAFE
        
    def get_nearest_obstacle_telemetry(self):
        """Finds nearest obstacle, returns relative vector and distance to surface."""
        min_dist = 99.0
        nearest_vector = (0.0, 0.0, 0.0)
        for cx, cy, cz, rad in self.obstacles:
            dx = cx - self.x
            dy = cy - self.y
            dz = cz - self.z
            dist_to_center = math.sqrt(dx**2 + dy**2 + dz**2)
            dist_to_surface = max(0.0, dist_to_center - rad)
            if dist_to_surface < min_dist:
                min_dist = dist_to_surface
                nearest_vector = (dx, dy, dz)
        return nearest_vector[0], nearest_vector[1], nearest_vector[2], min_dist
        
    def get_state_prompt(self):
        odx, ody, odz, obs_dist = self.get_nearest_obstacle_telemetry()
        return f"POS: {self.x:.1f},{self.y:.1f},{self.z:.1f} | VEL: {self.vx:.1f},{self.vy:.1f},{self.vz:.1f} | WIND: {self.wx:.1f},{self.wy:.1f},{self.wz:.1f} | OBS: {odx:.1f},{ody:.1f},{odz:.1f},{obs_dist:.1f} | BAT: {self.battery}% -> CMD:"

    def step(self, command):
        self.steps += 1
        self.battery -= 1 # Base drain
        
        # Parse command strictly
        clean_cmd = command.strip().upper()
        if clean_cmd not in VALID_COMMANDS_3D:
            self.battery -= 2 # Penalty for invalid syntax
            # Apply passive wind drift and inertia only
            self.x += self.vx + self.wx * 0.1
            self.y += self.vy + self.wy * 0.1
            self.z += self.vz + self.wz * 0.1
            return
            
        # Adjust velocity based on command
        accel = 1.0
        if clean_cmd == "FLY_FORWARD":
            # Direct heading vector towards target
            tx, ty, tz = self.target_x, self.target_y, self.target_z
            dx, dy, dz = tx - self.x, ty - self.y, tz - self.z
            mag = math.sqrt(dx**2 + dy**2 + dz**2)
            if mag > 0:
                self.vx = (dx / mag) * accel
                self.vy = (dy / mag) * accel
        elif clean_cmd == "FLY_UP":
            self.vz = 0.8
            self.vx *= 0.8
            self.vy *= 0.8
        elif clean_cmd == "FLY_DOWN":
            self.vz = -0.8
            self.vx *= 0.8
            self.vy *= 0.8
        elif clean_cmd == "FLY_LEFT":
            # Perpendicular steering
            self.vx = -0.7
            self.vy = 0.5
            self.vz *= 0.5
        elif clean_cmd == "FLY_RIGHT":
            # Perpendicular steering
            self.vx = 0.7
            self.vy = -0.5
            self.vz *= 0.5
        elif clean_cmd == "RETURN_TO_BASE":
            # Direct velocity to (0, 0, 2)
            dx, dy, dz = 0.0 - self.x, 0.0 - self.y, 2.0 - self.z
            mag = math.sqrt(dx**2 + dy**2 + dz**2)
            if mag > 0:
                self.vx = (dx / mag) * accel
                self.vy = (dy / mag) * accel
                self.vz = (dz / mag) * accel
        elif clean_cmd == "LAND":
            self.vx = 0.0
            self.vy = 0.0
            self.vz = -0.5
            
        # Apply physics update: pos = pos + vel + wind_drift
        self.x += self.vx + self.wx * 0.1
        self.y += self.vy + self.wy * 0.1
        self.z += self.vz + self.wz * 0.1
        
        # Bound altitude
        if self.z <= 0.0:
            self.z = 0.0
            if clean_cmd == "LAND" or self.vz < 0.2:
                self.status = "LANDED_SAFE"
            else:
                self.status = "CRASHED" # Hard ground collision
            return
            
        if self.battery <= 0:
            self.status = "OUT_OF_BATTERY"
            return
            
        # Collision detection (3D sphere check)
        for cx, cy, cz, rad in self.obstacles:
            dist = math.sqrt((self.x - cx)**2 + (self.y - cy)**2 + (self.z - cz)**2)
            if dist < rad:
                self.status = "CRASHED"
                return
                
        # Reached Target Check (within 1.5m sphere)
        dist_to_tgt = math.sqrt((self.x - self.target_x)**2 + (self.y - self.target_y)**2 + (self.z - self.target_z)**2)
        if dist_to_tgt <= 1.5:
            self.status = "REACHED_TARGET"

def evaluate_flight_3d(model, tokenizer, device, target_x=6.0, target_y=6.0, target_z=12.0, obstacles=None):
    """Runs a 3D closed-loop simulation flight using the CMF model."""
    sim = DroneSimulator3D(target_x, target_y, target_z, obstacles)
    
    print(f"\n>>> Starting 3D Flight Test | Target: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
    print(f"    Obstacles: {sim.obstacles} | Wind Vector: ({sim.wx:.1f}, {sim.wy:.1f}, {sim.wz:.1f})")
    
    max_steps = 30
    for s in range(max_steps):
        prompt = sim.get_state_prompt()
        prompt_x = tokenizer.encode(prompt).unsqueeze(0).to(device)
        
        # Autoregressive generation
        with torch.no_grad():
            gen_tokens = model.generate(
                prompt_x,
                max_new_tokens=12,
                temperature=0.01,
                use_velocity_halting=False
            )
            raw_gen = tokenizer.decode(gen_tokens[0, prompt_x.size(1):]).strip()
            
        # Strict parsing
        predicted_cmd = "INVALID"
        for cmd in VALID_COMMANDS_3D:
            if raw_gen.startswith(cmd):
                predicted_cmd = cmd
                break
                
        prev_pos = (sim.x, sim.y, sim.z)
        sim.step(predicted_cmd)
        
        print(f"  Step {sim.steps:02d} | Pos: ({prev_pos[0]:.1f}, {prev_pos[1]:.1f}, {prev_pos[2]:.1f}) -> {predicted_cmd} -> {sim.status} (Pos: ({sim.x:.1f}, {sim.y:.1f}, {sim.z:.1f}), Bat: {sim.battery}%)")
        
        if sim.status != "FLYING":
            break
            
    print(f"3D Flight Finished! Status: {sim.status} in {sim.steps} steps.")
    return sim.status

def run_3d_simulation_suite():
    print("======================================================================")
    print("             CMF 3D CLOSED-LOOP DRONE FLIGHT SIMULATOR                 ")
    print("======================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}\n")

    # 1. Setup CMF Config for a 120M Model
    config = CMFConfig(
        vocab_size=256,
        d_model=128,
        hidden_dim=256,
        num_layers=3,
        max_seq_len=128,
        thinking_steps=8,
        adaptive_thinking=True,
        min_thinking_steps=2,
        max_thinking_steps=8,
    )
    
    tokenizer = ByteTokenizer()
    model = DeliberativeContinuousMeaningField(config).to(device)
    
    # 2. Generate training data (3D state space)
    print(">>> Generating 3D state-action flight trajectories...")
    train_data = generate_flight_data_3d(num_samples=600)
    tokenized_data = []
    for item in train_data:
        p_t = tokenizer.encode(item["prompt"])
        a_t = tokenizer.encode(item["action"])
        full = torch.cat([p_t, a_t])
        labels = full.clone()
        labels[:len(p_t)] = -100
        tokenized_data.append({"input_ids": full, "labels": labels})

    # 3. Train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    model.train()
    
    print("\n>>> Training 3D Autopilot Network on 600 flight state trajectories...")
    batch_size = 24
    epochs = 80
    
    for epoch in range(epochs):
        random.shuffle(tokenized_data)
        epoch_loss = 0
        
        # Batch processing
        for i in range(0, len(tokenized_data), batch_size):
            batch = tokenized_data[i:i+batch_size]
            if not batch: continue
            
            max_len = max(len(x["input_ids"]) for x in batch)
            inputs_padded = []
            labels_padded = []
            for item in batch:
                pad_len = max_len - len(item["input_ids"])
                inputs_padded.append(F.pad(item["input_ids"], (0, pad_len), value=0))
                labels_padded.append(F.pad(item["labels"], (0, pad_len), value=-100))
                
            x = torch.stack(inputs_padded).to(device)
            y = torch.stack(labels_padded).to(device)
            
            optimizer.zero_grad()
            out = model(x, labels=y)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:02d}/{epochs:02d} | Training Loss: {epoch_loss / (len(tokenized_data)/batch_size):.4f}")

    # 4. Evaluate on Unseen 3D Flight Scenarios
    model.eval()
    
    # Flight Test 1: Obstacle blocking path directly (Obstacle at 3,3,10 with target at 6,6,12)
    # Drone starts at (0, 0, 10). It must fly UP to cross over the obstacle, then move forward.
    print("\n------------------ FLIGHT TEST 1: 3D OBSTACLE AVOIDANCE ------------------")
    evaluate_flight_3d(
        model, tokenizer, device,
        target_x=6.0, target_y=6.0, target_z=12.0,
        obstacles=[(3.0, 3.0, 10.0, 1.8)] # Obstacle directly in path
    )
    
    # Flight Test 2: Emergency low battery return and land
    print("\n------------------ FLIGHT TEST 2: EMERGENCY LOW BATTERY ------------------")
    sim2 = DroneSimulator3D(target_x=6.0, target_y=6.0, target_z=10.0)
    sim2.battery = 15 # Low battery start
    
    for s in range(30):
        prompt = sim2.get_state_prompt()
        prompt_x = tokenizer.encode(prompt).unsqueeze(0).to(device)
        with torch.no_grad():
            gen_tokens = model.generate(prompt_x, max_new_tokens=10, temperature=0.01, use_velocity_halting=False)
            raw_gen = tokenizer.decode(gen_tokens[0, prompt_x.size(1):]).strip()
            
        predicted_cmd = "INVALID"
        for cmd in VALID_COMMANDS_3D:
            if raw_gen.startswith(cmd):
                predicted_cmd = cmd
                break
                
        prev_pos = (sim2.x, sim2.y, sim2.z)
        sim2.step(predicted_cmd)
        print(f"  Step {sim2.steps:02d} | Telemetry: {prompt} -> Predicted: {predicted_cmd} -> State: {sim2.status} (Pos: ({sim2.x:.1f},{sim2.y:.1f},{sim2.z:.1f}), Bat: {sim2.battery}%)")
        if sim2.status != "FLYING":
            break
            
    print(f"3D Flight 2 Finished! Status: {sim2.status} in {sim2.steps} steps.")

if __name__ == "__main__":
    run_3d_simulation_suite()
