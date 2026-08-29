import numpy as np
from numba import njit, prange

# 16-State Voltage Logic Constants (0.0625V Steps)
STATE_SAFE_IDLE  = 0.0000  # 0.0V
STATE_SCANNING   = 0.0625  # 0.0625V
STATE_LOCK       = 0.5000  # 0.5V
STATE_FULL_DEST  = 1.0000  # 1.0V (Full Power to Target)

@njit(parallel=True, fastmath=True)
def execute_elemental_targeting_matrix(
    entity_positions,      # Shape: (N, 3) -> [X, Y, Z] Coordinate Vectors
    entity_masses,         # Shape: (N,) -> Quantified weight from Master P-Table calculations
    entity_volumes,        # Shape: (N,) -> Physical volumetric space occupied
    entity_dipoles,        # Shape: (N,) -> Quantified physical separation of electric charges
    entity_types,          # Shape: (N,) -> 1: Center, 2: Antenna, 3: Standard, 4: Active Weapons Systems
    weapon_positions,      # Shape: (W, 3) -> Hardpoint Vector Locations on Mobile Suit Frame
    close_fire_threshold,  # Float: Distance to initiate immediate kinetic dumping
    base_charge_constant   # Float: Calibrated background electrostatic reference
):
    """
    GUNDAM ROBOTICS SYSTEMS // CORE WEAPON ENGAGEMENT KERNEL
    INTEGRATED ELEMENTAL IDENTIFICATION & MATERIAL SEGREGATION MATRIX
    Compiles with parallel multi-thread core scaling over AlmaLinux 9.x layers.
    """
    num_entities = entity_positions.shape[0]
    num_weapons = weapon_positions.shape[0]
    
    weapon_assignments = np.full(num_weapons, -1, dtype=np.int32)
    fire_commands = np.zeros(num_weapons, dtype=np.int32)
    voltage_bus_states = np.full(num_entities, STATE_SCANNING, dtype=np.float64)
    
    # STEP 1: CALCULATE THE ATOMIC THREAT DENSITY & NET CHARGE PROFILE
    # Incorporates dipole moment molecular matching variables to isolate tracking targets
    calculated_threat_weights = np.zeros(num_entities, dtype=np.float64)
    for i in prange(num_entities):
        # Calculate material density from the Master P-Table metrics (Mass/Volume)
        material_density = entity_masses[i] / (entity_volumes[i] + 1e-9)
        
        # Quantify active polarization index from charge separation values
        polarization_index = np.abs(entity_dipoles[i] - base_charge_constant)
        
        # ENFORCE WEAPONS SYSTEMS AS MANDATORY ABSOLUTE FIRST PRIORITY
        if entity_types[i] == 4:     # Active Weapons Systems (Highest Allocation Bias)
            calculated_threat_weights[i] = 5000.0 * material_density * polarization_index
            voltage_bus_states[i] = STATE_LOCK
        elif entity_types[i] == 1:   # Structural Center Core
            calculated_threat_weights[i] = 50.0 * material_density
            voltage_bus_states[i] = STATE_LOCK
        elif entity_types[i] == 2:   # Communication Antenna Hubs
            calculated_threat_weights[i] = 10.0 * polarization_index
            voltage_bus_states[i] = STATE_LOCK
        else:                        # Standard Elements / Generic Debris
            calculated_threat_weights[i] = 1.0 * material_density

    # STEP 2: ASSIGN WEAPONS VIA INTEGRATED PROXIMITY AND TRACKING DENSITY CORES
    for w in range(num_weapons):
        w_pos = weapon_positions[w]
        min_dist = 1e12
        assigned_entity_idx = -1
        
        for e in range(num_entities):
            dx = entity_positions[e, 0] - w_pos
            dy = entity_positions[e, 1] - w_pos
            dz = entity_positions[e, 2] - w_pos
            actual_distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Massive priority multiplier coordinates weapon locks directly to enemy weapon layouts
            weighted_distance = actual_distance / (calculated_threat_weights[e] + 1e-9)
            
            if weighted_distance < min_dist:
                min_dist = weighted_distance
                assigned_entity_idx = e
                
        weapon_assignments[w] = assigned_entity_idx
        
        # STEP 3: FIRE OVERRIDE INJECTION AT MAXIMUM VOLTAGE STEP CAPACITY
        if assigned_entity_idx != -1:
            e_idx = assigned_entity_idx
            dx = entity_positions[e_idx, 0] - w_pos
            dy = entity_positions[e_idx, 1] - w_pos
            dz = entity_positions[e_idx, 2] - w_pos
            true_range = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Trigger full destruct output if target is an active weapon or within range
            if entity_types[e_idx] == 4 or true_range <= close_fire_threshold:
                fire_commands[w] = 1
                voltage_bus_states[e_idx] = STATE_FULL_DEST  # PUSH MAXIMUM VOLTAGE STEP
                
    return weapon_assignments, fire_commands, voltage_bus_states
