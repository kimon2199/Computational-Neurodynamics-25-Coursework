from iznetwork import IzNetwork
import numpy as np 

def generate_modular_network(p, params):
    """
    Generates a modular Izhikevich network for a given rewiring probability p.
    """
    
    # Unpack base parameters
    NUMBER_OF_MODULES = params['NUMBER_OF_MODULES']
    EXCITATORY_PER_MODULE = params['EXCITATORY_PER_MODULE']
    INHIBITORY_NEURONS = params['INHIBITORY_NEURONS']
    CONNECTIONS_PER_MODULE = params['CONNECTIONS_PER_MODULE']
    
    # Calculate derived parameters inside the function
    TOTAL_EXCITATORY_NEURONS = NUMBER_OF_MODULES * EXCITATORY_PER_MODULE
    TOTAL_NEURONS = TOTAL_EXCITATORY_NEURONS + INHIBITORY_NEURONS
    
    # === Initialize Network Structures ===
    
    # Izhikevich Neuron Parameters
    a = np.zeros(TOTAL_NEURONS)
    b = np.zeros(TOTAL_NEURONS)
    c = np.zeros(TOTAL_NEURONS)
    d = np.zeros(TOTAL_NEURONS)
    
    # Set up Izhikevich Parameters for Excitatory Neurons
    a[:TOTAL_EXCITATORY_NEURONS] = params['excitatory_iz_neuron']['a']
    b[:TOTAL_EXCITATORY_NEURONS] = params['excitatory_iz_neuron']['b']
    c[:TOTAL_EXCITATORY_NEURONS] = params['excitatory_iz_neuron']['c']
    d[:TOTAL_EXCITATORY_NEURONS] = params['excitatory_iz_neuron']['d']
    
    # Set up Izhikevich Parameters for Inhibitory Neurons
    a[TOTAL_EXCITATORY_NEURONS:] = params['inhibitory_iz_neuron']['a']
    b[TOTAL_EXCITATORY_NEURONS:] = params['inhibitory_iz_neuron']['b']
    c[TOTAL_EXCITATORY_NEURONS:] = params['inhibitory_iz_neuron']['c']
    d[TOTAL_EXCITATORY_NEURONS:] = params['inhibitory_iz_neuron']['d']
    
    # Connectivity and Delay Matrices
    W = np.zeros((TOTAL_NEURONS, TOTAL_NEURONS))
    # Initialize all delays to 1ms (since it is the default for 
    # E-I, I-E and I-I connections)
    # E-E delays will be overwritten
    D = np.ones((TOTAL_NEURONS, TOTAL_NEURONS), dtype=int) 

    
    # === Generate E-E Connections and Rewire ===
    # This is an implementation of the algorithm described in Lecture 4 Topic 8
    # using the weight, scaling factor and delay parameters from Topic 9
    
    print(f"  Generating E-E connections for p={p}...")
    
    # --- Step 1: Create all Intra-Modular Connections ---
    # (This is an implementation of Topic 8, Slide 3)
    print(f"    Step 1: Creating {CONNECTIONS_PER_MODULE * NUMBER_OF_MODULES} intra-modular connections...")
    ee_connections_list = [] # List to store connections for rewiring
    
    for i_mod in range(NUMBER_OF_MODULES):
        mod_start = i_mod * EXCITATORY_PER_MODULE
        mod_end = (i_mod + 1) * EXCITATORY_PER_MODULE
        
        connection_count = 0
        while connection_count < CONNECTIONS_PER_MODULE:
            src_node = np.random.randint(mod_start, mod_end)
            tgt_node = np.random.randint(mod_start, mod_end)
            
            if src_node == tgt_node or W[src_node, tgt_node] != 0: # Check for self or existing
                continue
            
            # Create the E-E connection
            # Numbers from Topic 9 Slide 12
            scaling_factor = 17
            weight = 1
            min_delay = 1
            max_delay = 20
            W[src_node, tgt_node] = scaling_factor * weight
            D[src_node, tgt_node] = np.random.randint(min_delay, max_delay + 1)
            
            # Store this connection to be (potentially) rewired
            ee_connections_list.append((src_node, tgt_node))
            connection_count += 1
            
    total_ee_conns = np.count_nonzero(W[:TOTAL_EXCITATORY_NEURONS, :TOTAL_EXCITATORY_NEURONS])
    print(f"    {total_ee_conns} E-E connections created.")

    # --- Step 2: Rewire connections based on p ---
    # (This is an implementation of Topic 8, Slide 4)
    print(f"    Step 2: Rewiring connections with p={p}...")
    rewired_count = 0
    
    for src_node, tgt_node in ee_connections_list:
        # Decide if this connection should be rewired
        if np.random.rand() < p:
            
            # --- Find a new Inter-modular connection ---
            new_tgt_found = False
            i_mod = src_node // EXCITATORY_PER_MODULE # Module of the source
            

            for _ in range(100): # 100 attempts to find a new target
                new_tgt = np.random.randint(0, TOTAL_EXCITATORY_NEURONS)
                new_mod_idx = new_tgt // EXCITATORY_PER_MODULE
                
                # Check validity:
                # 1. Not a self-connection
                # 2. Not in the same module as the source
                # 3. Connection doesn't already exist
                if (src_node != new_tgt and 
                    new_mod_idx != i_mod and 
                    W[src_node, new_tgt] == 0):
                    
                    # Create the new inter-modular connection
                    W[src_node, new_tgt] = 17.0 # Set E-E weight
                    D[src_node, new_tgt] = np.random.randint(1, 21) # Set E-E delay
                    
                    # Delete the old intra-modular connection
                    W[src_node, tgt_node] = 0.0
                    D[src_node, tgt_node] = 1 # Reset delay to default
                    
                    rewired_count += 1
                    new_tgt_found = True
                    break # Found a valid new target
            
            # If we failed to find a new target after 100 tries,
            # we just leave the original intra-modular connection.
            # (The rewire attempt fails, but total connections remain)
            if not new_tgt_found:
                # This print is for debugging, can be removed
                # print(f"    Warning: Could not find rewire target for {src} -> {tgt}")
                pass 
    
    print(f"    {rewired_count} connections were rewired.")

    
    # === 3. Generate Inhibitory Connections (Topic 9) ===
    
    print("  Generating inhibitory connections...")
    
    # A. E-I Connections (Focal)
    # 200 inh neurons, 8 modules -> 25 inh neurons per module //TODO:maybe random
    inh_per_module = INHIBITORY_NEURONS // NUMBER_OF_MODULES 
    for i_inh in range(INHIBITORY_NEURONS):
        inh_neuron_idx = TOTAL_EXCITATORY_NEURONS + i_inh
        # Find which module this inh neuron belongs to
        module_idx = i_inh // inh_per_module
        
        mod_start = module_idx * EXCITATORY_PER_MODULE
        mod_end = (module_idx + 1) * EXCITATORY_PER_MODULE
        
        # Randomly choose 4 unique source neurons from this module
        # (as per Topic 9)
        src_list = np.random.choice(range(mod_start, mod_end), 4, replace=False)
        
        for src_node in src_list:
            # Set weight (rand(0,1) * 50)
            W[src_node, inh_neuron_idx] = np.random.rand() * 50.0 # <-- Use W
            # Set delay (1ms)
            D[src_node, inh_neuron_idx] = 1 

    # B. I-E Connections (Diffuse)
    # (as per "Topic 9, Slide 1")
    for src_inh_idx in range(TOTAL_EXCITATORY_NEURONS, TOTAL_NEURONS):
        for tgt_exc_idx in range(TOTAL_EXCITATORY_NEURONS):
            # Set weight (rand(-1,0) * 2)
            W[src_inh_idx, tgt_exc_idx] = (np.random.rand() - 1.0) * 2.0
            # Set delay (1ms)
            D[src_inh_idx, tgt_exc_idx] = 1

    # C. I-I Connections (Diffuse)
    # (as per "Topic 9, Slide 1")
    for src_inh_idx in range(TOTAL_EXCITATORY_NEURONS, TOTAL_NEURONS):
        for tgt_inh_idx in range(TOTAL_EXCITATORY_NEURONS, TOTAL_NEURONS):
            if src_inh_idx != tgt_inh_idx:
                # Set weight (Topic 9, Slide 4: rand(-1,0) * 1)
                W[src_inh_idx, tgt_inh_idx] = np.random.rand() - 1.0
                # Set delay (Topic 9, Slide 4: 1ms)
                D[src_inh_idx, tgt_inh_idx] = 1

    # === 4. Finalizing the Network ===
    
    print("  Configuring IzNetwork instance...")
    
    # Dmax is 20ms from the E-E connections (Topic 9, Slide 4)
    network = IzNetwork(N=TOTAL_NEURONS, Dmax=20)
    network.setParameters(a, b, c, d)
    network.setWeights(W)
    network.setDelays(D)
    
    print(f"  Network for p={p} generated.")
    return network


# === Global Parameters ===
network_params = {
    # Modular Networks Experimental Setup (from Lecture 4 Topic 9)
    'NUMBER_OF_MODULES': 8,
    'EXCITATORY_PER_MODULE': 100,
    'INHIBITORY_NEURONS': 200,
    'CONNECTIONS_PER_MODULE': 1000,
    
    # Excitatory Izhikevich Neuron parameters (from Lecture 2 Topic 4)
    'excitatory_iz_neuron': {
        'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0
    },
    
    # Inhibitory Izhikevich Neuron parameters (from Lecture 2 Topic 4)
    'inhibitory_iz_neuron': {
        'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 2.0
    }
}

# The p values specified in coursework
P_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Dictionary to store the generated networks
networks = {}

# === 2. Main Loop ===
print("--- Starting network generation ---")

for p in P_VALUES:
    print(f"--- Generating network for p = {p} ---")
    
    net = generate_modular_network(p, network_params)
    networks[p] = net
    
    print(f"--- Finished p = {p} ---")

print(f"--- All 6 networks generated ---")