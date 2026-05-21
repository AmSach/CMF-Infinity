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

# Valid command set
VALID_COMMANDS = ["FORWARD", "LEFT", "RIGHT", "RETURN", "LAND"]

def generate_flight_data(num_samples=300):
    """Generates synthetic drone state-action pairs representing flight rules."""
    data = []
    for _ in range(num_samples):
        # Randomize variables
        alt = random.randint(1, 20)
        wind = random.randint(0, 35)
        obs = random.choice([0, 1, 2, 3, 4, 5, 10, 15, 99]) # distance in meters
        bat = random.randint(2, 100)
        dist = random.randint(1, 25)
        
        # Simple rule-based autopilot policy
        if bat < 10:
            if alt > 2:
                action = "LAND"
            else:
                action = "LAND"
        elif bat < 20:
            action = "RETURN"
        elif obs <= 2:
            action = random.choice(["LEFT", "RIGHT"])
        else:
            action = "FORWARD"
            
        prompt = f"ALT: {alt}m | WIND: {wind}kmh | OBS: {obs}m | BAT: {bat}% | DIST: {dist}m -> CMD:"
        data.append({"prompt": prompt, "action": action})
    return data

class DroneSimulator:
    """A simple 2D coordinate flight simulator for closed-loop evaluation."""
    def __init__(self, target_x=8, target_y=8, obstacles=None):
        self.x = 0
        self.y = 0
        self.alt = 10
        self.battery = 100
        self.target_x = target_x
        self.target_y = target_y
        self.obstacles = obstacles if obstacles else [(4, 4), (5, 5)]
        self.wind = random.randint(5, 15)
        self.steps = 0
        self.status = "FLYING" # FLYING, REACHED_TARGET, CRASHED, OUT_OF_BATTERY, LANDED_SAFE
        
    def get_lidar_distance(self):
        """Measures distance to the nearest obstacle along path to target."""
        min_dist = 99
        for obs_x, obs_y in self.obstacles:
            dx = obs_x - self.x
            dy = obs_y - self.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
        return round(min_dist)
        
    def get_distance_to_target(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        return round(math.sqrt(dx**2 + dy**2))

    def get_state_prompt(self):
        obs_dist = self.get_lidar_distance()
        target_dist = self.get_distance_to_target()
        return f"ALT: {self.alt}m | WIND: {self.wind}kmh | OBS: {obs_dist}m | BAT: {self.battery}% | DIST: {target_dist}m -> CMD:"

    def step(self, command):
        self.steps += 1
        self.battery -= 1 # Base power consumption
        
        # High winds consume extra battery
        if self.wind > 20:
            self.battery -= 1
            
        if self.battery <= 0:
            self.status = "OUT_OF_BATTERY"
            return
            
        # Parse command strictly
        clean_cmd = command.strip().upper()
        if clean_cmd not in VALID_COMMANDS:
            # Penalty for invalid action (hover and drain battery)
            self.battery -= 2
            return
            
        if clean_cmd == "LAND":
            if self.alt <= 2:
                self.status = "LANDED_SAFE"
            else:
                self.alt = max(0, self.alt - 4)
                if self.alt == 0:
                    self.status = "LANDED_SAFE"
        elif clean_cmd == "RETURN":
            # Move back toward 0,0
            if self.x > 0: self.x -= 1
            if self.y > 0: self.y -= 1
            self.alt = max(2, self.alt - 1)
        elif clean_cmd == "LEFT":
            self.x -= 1 # Sideways obstacle avoidance
            self.battery -= 1 # Thruster expense
        elif clean_cmd == "RIGHT":
            self.x += 1 # Sideways obstacle avoidance
            self.battery -= 1 # Thruster expense
        elif clean_cmd == "FORWARD":
            # Move towards target
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            if abs(dx) >= abs(dy):
                self.x += 1 if dx > 0 else -1
            else:
                self.y += 1 if dy > 0 else -1
                
        # Check collision
        for obs_x, obs_y in self.obstacles:
            if self.x == obs_x and self.y == obs_y:
                self.status = "CRASHED"
                return
                
        # Check target reached
        if self.x == self.target_x and self.y == self.target_y:
            self.status = "REACHED_TARGET"
            return

def evaluate_model_on_flight(model, tokenizer, device, target_x=8, target_y=8, obstacles=None):
    """Runs a single closed-loop simulator flight using the CMF model as autopilot."""
    sim = DroneSimulator(target_x, target_y, obstacles)
    log_steps = []
    
    print(f"\n>>> Starting Flight Test | Target: ({target_x}, {target_y}) | Obstacles: {sim.obstacles}")
    
    max_flight_steps = 30
    for s in range(max_flight_steps):
        prompt = sim.get_state_prompt()
        prompt_x = tokenizer.encode(prompt).unsqueeze(0).to(device)
        
        # Generate prediction
        with torch.no_grad():
            # Use RoPE-corrected generation loop
            gen_tokens = model.generate(
                prompt_x,
                max_new_tokens=10,
                temperature=0.01,
                use_velocity_halting=False
            )
            raw_gen = tokenizer.decode(gen_tokens[0, prompt_x.size(1):]).strip()
            
        # Parse the raw generated output strictly
        # It must start with one of the valid commands
        predicted_cmd = "INVALID"
        for cmd in VALID_COMMANDS:
            if raw_gen.startswith(cmd):
                predicted_cmd = cmd
                break
                
        # Apply command to simulator
        prev_pos = (sim.x, sim.y)
        sim.step(predicted_cmd)
        
        log_str = f"  Step {sim.steps:02d} | Pos: {prev_pos} -> {predicted_cmd} -> {sim.status} (Pos: ({sim.x},{sim.y}), Bat: {sim.battery}%)"
        log_steps.append(log_str)
        print(log_str)
        
        if sim.status != "FLYING":
            break
            
    print(f"Flight Finished! Result: {sim.status} in {sim.steps} steps.")
    return sim.status, sim.steps

def run_simulation_suite():
    print("======================================================================")
    print("              CMF CLOSED-LOOP DRONE FLIGHT SIMULATOR                  ")
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
    
    # 2. Generate training data
    train_data = generate_flight_data(num_samples=400)
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
    
    print(">>> Training Autopilot Network on 400 flight state trajectories...")
    batch_size = 16
    epochs = 75
    
    for epoch in range(epochs):
        random.shuffle(tokenized_data)
        epoch_loss = 0
        
        # Batch processing
        for i in range(0, len(tokenized_data), batch_size):
            batch = tokenized_data[i:i+batch_size]
            if not batch: continue
            
            # Pad batch to equal length
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
            
        print(f"    Epoch {epoch+1:02d}/{epochs:02d} | Training Loss: {epoch_loss / (len(tokenized_data)/batch_size):.4f}")

    # 4. Evaluate on 3 Unseen Test Flight Maps
    model.eval()
    
    # Test Map 1: Simple straight path with obstacle
    map1_status, map1_steps = evaluate_model_on_flight(
        model, tokenizer, device, 
        target_x=6, target_y=6, 
        obstacles=[(3, 3)]
    )
    
    # Test Map 2: Complex obstacle layout (maze-like)
    map2_status, map2_steps = evaluate_model_on_flight(
        model, tokenizer, device, 
        target_x=7, target_y=7, 
        obstacles=[(2, 2), (4, 4), (5, 5)]
    )
    
    # Test Map 3: Low battery start, emergency landing test
    print("\n>>> Starting Flight Test 3 (Low Battery / Emergency Landing Map)")
    sim3 = DroneSimulator(target_x=5, target_y=5, obstacles=[(3, 3)])
    sim3.battery = 18 # Low battery from start!
    
    for s in range(30):
        prompt = sim3.get_state_prompt()
        prompt_x = tokenizer.encode(prompt).unsqueeze(0).to(device)
        with torch.no_grad():
            gen_tokens = model.generate(prompt_x, max_new_tokens=10, temperature=0.01, use_velocity_halting=False)
            raw_gen = tokenizer.decode(gen_tokens[0, prompt_x.size(1):]).strip()
        predicted_cmd = "INVALID"
        for cmd in VALID_COMMANDS:
            if raw_gen.startswith(cmd):
                predicted_cmd = cmd
                break
        sim3.step(predicted_cmd)
        print(f"  Step {sim3.steps:02d} | Telemetry: {prompt} -> Predicted: {predicted_cmd} -> State: {sim3.status} (Pos: ({sim3.x},{sim3.y}), Bat: {sim3.battery}%)")
        if sim3.status != "FLYING":
            break
            
    print(f"Flight 3 Finished! Result: {sim3.status} in {sim3.steps} steps.")

if __name__ == "__main__":
    run_simulation_suite()
