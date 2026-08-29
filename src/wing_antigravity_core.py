import numpy as np
from numba import njit, prange

# 16-State Voltage Logic Constants (0.0625V Steps)
STATE_SAFE_IDLE  = 0.0000  # 0.0V
STATE_SCANNING   = 0.0625  # 0.0625V
STATE_LOCK       = 0.5000  # 0.5V
STATE_FULL_DEST  = 1.0000  # 1.0V (Full Power / Full Repulsion Charge)

@njit(parallel=True, fastmath=True)
def execute_wing_antigravity_matrix(
    plate_angles,          # Shape: (360,) -> Angular position of each segment [0-359]
    target_positions,      # Shape: (N, 3) -> [X, Y, Z] Tactical Entities
    target_types,          # Shape: (N,) -> 1: Center, 2: Antenna, 4: Active Weapons Systems
    current_velocity,      # Shape: (3,) -> [Vx, Vy, Vz] Flight vector
    vessel_mass,           # Float: Mass of the mobile suit (e.g., 1000.0 kg for hovering calculations)
    earth_field_intensity, # Float: Earth surface electric field reference (approx 100.0 N/C)
    turning_radius         # Float: Intended turn radius for sharp trajectory shifts
):
    """
    GUNDAM ROBOTICS SYSTEMS // TYPE-S WING ZERO FLIGHT DYNAMICS KERNEL
    MONOLITHIC CHARGE SEGMENTATION & OMNIDIRECTIONAL STEERING MATRIX
    Compiles with parallel multi-thread core scaling over AlmaLinux 9.x layers.
    """
    num_plates = plate_angles.shape[0]
    num_targets = target_positions.shape[0]
    
    # 1. CALCULATE BASELINE STABLE HOVER CHARGE EQUATION
    # Fg = m * g // Fe = q * E -> q = (m * g) / E
    gravity_acceleration = 9.8
    required_total_charge = (vessel_mass * gravity_acceleration) / earth_field_intensity # 98 Coulombs baseline
    base_charge_per_plate = required_total_charge / num_plates
    
    allocated_plate_charges = np.full(num_plates, base_charge_per_plate, dtype=np.float64)
    voltage_bus_states = np.full(num_plates, STATE_SCANNING, dtype=np.float64)
    
    # 2. EVALUATE WEAPON-FIRST PRIORITY TO ALTER CHARGE SHIFT BIAS
    primary_threat_vector = np.zeros(3, dtype=np.float64)
    highest_threat_weight = -1.0
    
    for t in prange(num_targets):
        if target_types[t] == 4:  # MANDATORY HIGHEST PRIORITY: Target Weapon Systems
            dx = target_positions[t, 0]
            dy = target_positions[t, 1]
            dz = target_positions[t, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            threat_weight = 1000.0 / (dist + 1e-9)
            
            if threat_weight > highest_threat_weight:
                highest_threat_weight = threat_weight
                primary_threat_vector[0] = dx
                primary_threat_vector[1] = dy
                primary_threat_vector[2] = dz

    # 3. OMNIDIRECTIONAL MANEUVERABILITY (THE ZIG-ZAG TRAJECTORY SHIFT)
    # If a weapon-first lock or hypersonic turn is triggered, modify charge densities
    v_magnitude = np.sqrt(current_velocity[0]**2 + current_velocity[1]**2 + current_velocity[2]**2)
    if v_magnitude > 0.0 and turning_radius > 0.0:
        # ac = v^2 / r
        centripetal_accel = (v_magnitude * v_magnitude) / turning_radius
        
        # Calculate leading vector offset toward the threat coordinate or turn direction
        target_angle_rad = np.arctan2(primary_threat_vector[1], primary_threat_vector[0])
        target_angle_deg = np.degrees(target_angle_rad) % 360.0
        
        # Shift charge density directly into the leading quadrant plates
        for p in range(num_plates):
            angle_diff = np.abs(plate_angles[p] - target_angle_deg)
            if angle_diff > 180.0:
                angle_diff = 360.0 - angle_diff
                
            if angle_diff <= 45.0:  # Leading 90-degree sector
                # Asymmetric repulsion vector injection
                charge_multiplier = 1.0 + (centripetal_accel / gravity_acceleration)
                allocated_plate_charges[p] *= charge_multiplier
                voltage_bus_states[p] = STATE_FULL_DEST
            elif angle_diff >= 135.0:  # Trailing quadrant plates
                allocated_plate_charges[p] *= 0.1
                voltage_bus_states[p] = STATE_SAFE_IDLE
                
    return allocated_plate_charges, voltage_bus_states
