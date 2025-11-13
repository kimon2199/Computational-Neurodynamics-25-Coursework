from src.iznetwork import IzNetwork
import numpy as np
import matplotlib.pyplot as plt
import os


class ModularNetwork:
    """
    Class to generate modular Izhikevich networks with specified rewiring
    probability p.
    """

    def __init__(self, p: float, params: dict):
        """
        Initializes the Modular Network Generator with rewiring probability p
        and network parameters.

        Parameters:
        p      -- Rewiring probability for E-E connections.
        params -- Dictionary containing network parameters.
        """
        self.p = p
        self.params = params

        # Unpack base parameters
        self.NUMBER_OF_MODULES = self.params["NUMBER_OF_MODULES"]
        self.EXCITATORY_PER_MODULE = self.params["EXCITATORY_PER_MODULE"]
        self.INHIBITORY_NEURONS = self.params["INHIBITORY_NEURONS"]
        self.CONNECTIONS_PER_MODULE = self.params["CONNECTIONS_PER_MODULE"]

        # Calculate derived parameters inside the function
        self.TOTAL_EXCITATORY_NEURONS = (
            self.NUMBER_OF_MODULES * self.EXCITATORY_PER_MODULE
        )
        self.TOTAL_NEURONS = self.TOTAL_EXCITATORY_NEURONS + self.INHIBITORY_NEURONS

        self.network: IzNetwork = None
        # Connectivity and Delay Matrices
        self.W = np.zeros((self.TOTAL_NEURONS, self.TOTAL_NEURONS))
        # Initialize all delays to 1ms (since it is the default for
        # E-I, I-E and I-I connections)
        # E-E delays will be overwritten
        self.D = np.ones((self.TOTAL_NEURONS, self.TOTAL_NEURONS), dtype=int)

    def _generate_neuron_parameters(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initializes the Izhikevich neuron parameters for the entire network.
        Excitatory and Inhibitory neurons have different parameters.

        Returns:
        a, b, c, d -- np.arrays of size TOTAL_NEURONS containing the parameters for each neuron.
        """
        # Izhikevich Neuron Parameters
        a = np.zeros(self.TOTAL_NEURONS)
        b = np.zeros(self.TOTAL_NEURONS)
        c = np.zeros(self.TOTAL_NEURONS)
        d = np.zeros(self.TOTAL_NEURONS)

        excitatory_r = np.random.rand(self.TOTAL_EXCITATORY_NEURONS)
        inhibitory_r = np.random.rand(self.INHIBITORY_NEURONS)

        # Set up Izhikevich Parameters for Excitatory Neurons (With Randomness)
        a[: self.TOTAL_EXCITATORY_NEURONS] = (
            self.params["excitatory_iz_neuron"]["a"]
            + self.params["excitatory_iz_neuron"]["a_r"] * excitatory_r**2
        )
        b[: self.TOTAL_EXCITATORY_NEURONS] = (
            self.params["excitatory_iz_neuron"]["b"]
            + self.params["excitatory_iz_neuron"]["b_r"] * excitatory_r**2
        )
        c[: self.TOTAL_EXCITATORY_NEURONS] = (
            self.params["excitatory_iz_neuron"]["c"]
            + self.params["excitatory_iz_neuron"]["c_r"] * excitatory_r**2
        )
        d[: self.TOTAL_EXCITATORY_NEURONS] = (
            self.params["excitatory_iz_neuron"]["d"]
            + self.params["excitatory_iz_neuron"]["d_r"] * excitatory_r**2
        )

        # Set up Izhikevich Parameters for Inhibitory Neurons (With Randomness)
        a[self.TOTAL_EXCITATORY_NEURONS :] = (
            self.params["inhibitory_iz_neuron"]["a"]
            + self.params["inhibitory_iz_neuron"]["a_r"] * inhibitory_r
        )
        b[self.TOTAL_EXCITATORY_NEURONS :] = (
            self.params["inhibitory_iz_neuron"]["b"]
            + self.params["inhibitory_iz_neuron"]["b_r"] * inhibitory_r
        )
        c[self.TOTAL_EXCITATORY_NEURONS :] = (
            self.params["inhibitory_iz_neuron"]["c"]
            + self.params["inhibitory_iz_neuron"]["c_r"] * inhibitory_r
        )
        d[self.TOTAL_EXCITATORY_NEURONS :] = (
            self.params["inhibitory_iz_neuron"]["d"]
            + self.params["inhibitory_iz_neuron"]["d_r"] * inhibitory_r
        )

        return a, b, c, d

    def _generate_ee_connections(self) -> list[tuple[int, int]]:
        """
        Generates all Intra-Modular E-E connections.
        This is an implementation of Topic 8, Slide 3.

        Returns:
        ee_connections_list -- List of tuples (src_node, tgt_node) for all created E-E connections.
        """
        ee_connections_list = []  # List to store connections for rewiring

        for i_mod in range(self.NUMBER_OF_MODULES):
            mod_start = i_mod * self.EXCITATORY_PER_MODULE
            mod_end = (i_mod + 1) * self.EXCITATORY_PER_MODULE

            connection_count = 0
            while connection_count < self.CONNECTIONS_PER_MODULE:
                src_node = np.random.randint(mod_start, mod_end)
                tgt_node = np.random.randint(mod_start, mod_end)

                if (
                    src_node == tgt_node or self.W[src_node, tgt_node] != 0
                ):  # Check for self or existing
                    continue

                # Create the E-E connection
                # Numbers from Topic 9 Slide 12
                scaling_factor = 17
                weight = 1
                min_delay = 1
                max_delay = 20
                self.W[src_node, tgt_node] = scaling_factor * weight
                self.D[src_node, tgt_node] = np.random.randint(min_delay, max_delay + 1)

                # Store this connection to be (potentially) rewired
                ee_connections_list.append((src_node, tgt_node))
                connection_count += 1

        return ee_connections_list

    def _rewire_ee_connections(self, ee_connections_list: list[tuple[int, int]]) -> int:
        """
        Rewires E-E connections based on the rewiring probability p.
        Returns the number of connections that were rewired.

        Parameters:
        ee_connections_list -- List of tuples (src_node, tgt_node) for all created E-E connections.

        Returns:
        rewired_count -- Number of connections that were successfully rewired.
        """

        rewired_count = 0

        for src_node, tgt_node in ee_connections_list:
            # Decide if this connection should be rewired
            if np.random.rand() < self.p:

                # --- Find a new Inter-modular connection ---
                new_tgt_found = False
                i_mod = src_node // self.EXCITATORY_PER_MODULE  # Module of the source

                for _ in range(100):  # 100 attempts to find a new target
                    new_tgt = np.random.randint(0, self.TOTAL_EXCITATORY_NEURONS)
                    new_mod_idx = new_tgt // self.EXCITATORY_PER_MODULE

                    # Check validity:
                    # 1. Not a self-connection
                    # 2. Not in the same module as the source
                    # 3. Connection doesn't already exist
                    if (
                        src_node != new_tgt
                        and new_mod_idx != i_mod
                        and self.W[src_node, new_tgt] == 0
                    ):

                        # Create the new inter-modular connection
                        self.W[src_node, new_tgt] = 17.0  # Set E-E weight
                        self.D[src_node, new_tgt] = np.random.randint(
                            1, 21
                        )  # Set E-E delay

                        # Delete the old intra-modular connection
                        self.W[src_node, tgt_node] = 0.0
                        self.D[src_node, tgt_node] = 1  # Reset delay to default

                        rewired_count += 1
                        new_tgt_found = True
                        break  # Found a valid new target

                # If we failed to find a new target after 100 tries,
                # we just leave the original intra-modular connection.
                # (The rewire attempt fails, but total connections remain)
                if not new_tgt_found:
                    # This print is for debugging, can be removed
                    # print(f"    Warning: Could not find rewire target for {src} -> {tgt}")
                    pass

        return rewired_count

    def _generate_ei_connections(self):
        """
        Creates Focal E-I connections.
        1. Every E-neuron sends exactly ONE connection (mentined on EdStem)
        2. Every I-neuron receives exactly FOUR connections (mentioned in Topic 9)
        """
        # 200 inh neurons, 8 modules -> 25 inh neurons per module
        inh_per_module = self.INHIBITORY_NEURONS // self.NUMBER_OF_MODULES

        # Loop through each module to create a perfect 100-to-25 mapping
        for module_idx in range(self.NUMBER_OF_MODULES):

            # --- Get the 100 E-neurons for this module ---
            mod_exc_start = module_idx * self.EXCITATORY_PER_MODULE
            mod_exc_end = (module_idx + 1) * self.EXCITATORY_PER_MODULE
            e_neurons_in_module = np.arange(mod_exc_start, mod_exc_end)

            # --- Get the 25 I-neurons for this module ---
            mod_inh_start = self.TOTAL_EXCITATORY_NEURONS + (
                module_idx * inh_per_module
            )
            mod_inh_end = self.TOTAL_EXCITATORY_NEURONS + (
                (module_idx + 1) * inh_per_module
            )
            i_neurons_in_module = np.arange(mod_inh_start, mod_inh_end)

            # --- Create a "target list" ---
            # This list will have 100 items (4 * 25).
            # Each of the 25 I-neurons appears exactly 4 times.
            target_list = np.repeat(i_neurons_in_module, 4)

            # --- Shuffle the target list ---
            # This randomizes which 4 E-neurons map to which I-neuron.
            np.random.shuffle(target_list)

            # --- Create the guaranteed connections ---
            # e_neurons_in_module[i] maps to target_list[i]
            for src_node, tgt_inh_idx in zip(e_neurons_in_module, target_list):
                # Create the connection
                # Set weight (rand(0,1) * 50)
                self.W[src_node, tgt_inh_idx] = np.random.rand() * 50.0
                # Set delay (1ms)
                self.D[src_node, tgt_inh_idx] = 1

    def _generate_ie_connections(self):
        """
        Creates Focal I-E connections as specified in Topic 9.
        """
        for src_inh_idx in range(self.TOTAL_EXCITATORY_NEURONS, self.TOTAL_NEURONS):
            for tgt_exc_idx in range(self.TOTAL_EXCITATORY_NEURONS):
                # Set weight (rand(-1,0) * 2)
                self.W[src_inh_idx, tgt_exc_idx] = (np.random.rand() - 1.0) * 2.0
                # Set delay (1ms)
                self.D[src_inh_idx, tgt_exc_idx] = 1

    def _generate_ii_connections(self):
        """
        Creates Focal I-I connections as specified in Topic 9.
        """
        for src_inh_idx in range(self.TOTAL_EXCITATORY_NEURONS, self.TOTAL_NEURONS):
            for tgt_inh_idx in range(self.TOTAL_EXCITATORY_NEURONS, self.TOTAL_NEURONS):
                if src_inh_idx != tgt_inh_idx:
                    # Set weight (Topic 9, Slide 4: rand(-1,0) * 1)
                    self.W[src_inh_idx, tgt_inh_idx] = np.random.rand() - 1.0
                    # Set delay (Topic 9, Slide 4: 1ms)
                    self.D[src_inh_idx, tgt_inh_idx] = 1

    def generate_modular_network(self) -> IzNetwork:
        """
        Generates a modular Izhikevich network for a given rewiring probability p.

        Returns:
        network -- An instance of IzNetwork configured with the generated connections.
        """
        # === Generate E-E Connections and Rewire ===
        # This is an implementation of the algorithm described in Lecture 4 Topic 8
        # using the weight, scaling factor and delay parameters from Topic 9
        print(f"Generating E-E connections for p={self.p}...")
        print(
            f"Creating {self.CONNECTIONS_PER_MODULE * self.NUMBER_OF_MODULES} intra-modular connections"
        )
        ee_connections_list = self._generate_ee_connections()

        total_ee_conns = np.count_nonzero(
            self.W[: self.TOTAL_EXCITATORY_NEURONS, : self.TOTAL_EXCITATORY_NEURONS]
        )
        print(f"{total_ee_conns} E-E connections created.")

        # --- Step 2: Rewire connections based on p ---
        # (This is an implementation of Topic 8, Slide 4)
        print(f"Step 2: Rewiring connections with p={self.p}...")
        rewired_count = self._rewire_ee_connections(ee_connections_list)

        print(f"{rewired_count} connections were rewired.")

        # === 3. Generate Inhibitory Connections (Topic 9) ===
        print("Generating inhibitory connections...")

        # A. E-I Connections (Focal)
        self._generate_ei_connections()

        # B. I-E Connections (Diffuse)
        self._generate_ie_connections()

        # C. I-I Connections (Diffuse)
        self._generate_ii_connections()

        # === 4. Finalizing the Network ===
        print("Configuring IzNetwork instance...")

        # Dmax is 20ms from the E-E connections (Topic 9, Slide 4)
        self.network = IzNetwork(N=self.TOTAL_NEURONS, Dmax=20)
        self.network.setParameters(*self._generate_neuron_parameters())
        self.network.setWeights(self.W)
        self.network.setDelays(self.D)

        print(f"Network for p={self.p} generated.")
        return self.network

    def run_simulation(self, sim_time: int) -> list[tuple[int, int]]:
        """
        Runs a simulation of the generated network for a specified time.

        Parameters:
        sim_time -- Simulation time in milliseconds.

        Returns:
        spikes -- List of spike times in ms and neuron indices: (t, idx).
        """
        if self.network is None:
            raise ValueError("Network has not been generated yet.")

        spikes = []
        for t in range(sim_time):

            # Set random current guided by a poisson process (Topic 9 slide 11)
            poisson_values = np.random.poisson(lam=0.01, size=self.TOTAL_NEURONS)
            input_current = np.where(poisson_values > 0, 15.0, 0.0)
            self.network.setCurrent(input_current)

            # Simulate 1ms of the network firings
            fired_neurons = self.network.update()

            for neuron_idx in fired_neurons:
                spikes.append((t, neuron_idx))

        return spikes

    def connectivity_matrix(
        self,
        excitatory_only: bool = False,
        title="Connection matrix",
        save_plot: bool = False,
        plot_filename: str = "connection_matrix.svg",
    ):
        """
        Visualize the binary connectivity of a neural network.

        Parameters
        ----------
        W : np.ndarray
            Synaptic weight matrix where W[i, j] represents the connection
            strength from neuron i → neuron j.
        title : str, optional
            Title of the plot (default = "Connection matrix").
        save_plot : bool
            If True, saves the plot as an SVG file.
        plot_filename : str
            Filename for saving the plot (default = "connection_matrix.svg").
        """
        if excitatory_only:
            n_exc = self.TOTAL_EXCITATORY_NEURONS
            W = self.W[:n_exc, :n_exc]  # strictly excitatory to excitatory
            binary_W = (W != 0).astype(int)
            title = f"{title} (excitatory only)"
        else:
            binary_W = (self.W != 0).astype(int)

        plt.figure(figsize=(6, 6))
        plt.spy(binary_W, markersize=0.2, color="black")
        plt.title(title, fontsize=14)
        plt.xlabel("Neuron j")
        plt.ylabel("Neuron i")
        plt.tight_layout()
        if save_plot:
            plt.savefig(plot_filename, format="svg")
            print(f"Connection matrix plot saved as {plot_filename}")
        # plt.show()

    def raster_plot(
        self,
        spikes: list[tuple[int, int]],
        sim_time: int = 1000,
        excitatory_only: bool = False,
        y0_on_top: bool = True,
        save_plot: bool = False,
        plot_filename: str = "raster_plot.svg",
    ):
        """
        Generate a raster plot of neuron firing from precomputed spike data.

        Parameters
        ----------
        spikes : list[tuple[int, int]]
            List of (time, neuron_index) tuples returned by run_simulation().
        sim_time : int
            Duration of the simulation in milliseconds (default: 1000 ms).
        excitatory_only : bool
            If True, only plot excitatory neurons.
        y0_on_top : bool
            If True, show neuron 0 at the top (reverse y-axis).
        save_plot : bool
            If True, saves the plot as an SVG file.
        plot_filename : str
            Filename for saving the plot (default = "raster_plot.svg").
        """
        if not spikes:
            print("No spikes provided — nothing to plot.")
            return

        # Unpack spike times and neuron indices
        spike_times, spike_neurons = zip(*spikes)
        spike_times = np.array(spike_times)
        spike_neurons = np.array(spike_neurons)

        # Filter excitatory neurons if requested
        if excitatory_only:
            mask = spike_neurons < self.TOTAL_EXCITATORY_NEURONS
            print(f"Excitatory spikes retained: {np.count_nonzero(mask)}")
            spike_times = spike_times[mask]
            spike_neurons = spike_neurons[mask]
            if spike_times.size == 0:
                print("No excitatory spikes to plot for the provided data.")
                return
            y_max = self.TOTAL_EXCITATORY_NEURONS
            title_extra = " (excitatory only)"
        else:
            y_max = self.TOTAL_NEURONS
            title_extra = ""

        # Plot
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.scatter(spike_times, spike_neurons, s=10, color="blue")
        plt.title(f"Raster plot (p = {self.p}){title_extra}", fontsize=12)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")

        # X limits
        ax.set_xlim(0, sim_time)

        # Y limits (optionally reversed so 0 is at top)
        if y0_on_top:
            ax.set_ylim(y_max - 1, 0)  # reversed
        else:
            ax.set_ylim(0, y_max - 1)  # normal

        plt.tight_layout()
        if save_plot:
            plt.savefig(plot_filename, format="svg")
            print(f"Raster plot saved as {plot_filename}")
        # plt.show()

    def mean_firing_rate(
        self,
        activations: list[tuple[int, int]],
        sim_time: int,
        window_size: int,
        step_size: int,
        include_inhibitory: bool = True,
        save_plot: bool = False,
        plot_filename: str = "mean_firing_rate.svg",
    ):
        """
        Plots the mean firing rate of the network.

        Parameters:
        activations -- List of spike times in ms and neuron indices: (t, idx).
        sim_time    -- Total simulation time in ms.
        window_size -- Size of the sliding window in ms.
        step_size   -- Step size for the sliding window in ms.
        include_inhibitory -- If True, includes inhibitory neurons in the plot.
        save_plot -- If True, saves the plot as an SVG file.
        plot_filename -- Filename for saving the plot (default = "mean_firing_rate.svg").
        """
        if self.network is None:
            raise ValueError("Network has not been generated yet.")

        # Number of modules + 1 for inhibitory neurons
        n_firings = np.zeros((self.NUMBER_OF_MODULES + 1, sim_time // step_size))

        for t, neuron_idx in activations:

            if neuron_idx >= self.TOTAL_EXCITATORY_NEURONS:
                # Inhibitory neuron
                module_idx = self.NUMBER_OF_MODULES  # Last index for inhibitory
            else:
                module_idx = neuron_idx // self.EXCITATORY_PER_MODULE

            # Get the time bins the spike falls into
            for bin_idx in range(
                max(0, (t - window_size) // step_size + 1),
                min(t // step_size + 1, sim_time // step_size),
            ):
                n_firings[module_idx, bin_idx] += 1

        # Plot the firing rates as a line plot
        time_bins = np.arange(0, sim_time, step_size)
        plt.figure(figsize=(12, 6))

        for module_idx in range(
            self.NUMBER_OF_MODULES + (1 if include_inhibitory else 0)
        ):
            firing_rates = n_firings[module_idx, :] / window_size  # firings per ms

            label = (
                f"Module {module_idx + 1}"
                if module_idx < self.NUMBER_OF_MODULES
                else "Inhibitory Neurons"
            )
            plt.plot(
                time_bins,
                firing_rates,
                label=label,
            )

        plt.xlabel("Time (ms)")
        plt.ylabel("Mean Firing Rate (firings/ms) per Neuron")
        plt.title(f"Mean Firing Rate Over Time (p = {self.p})")
        plt.legend()
        if save_plot:
            plt.savefig(plot_filename, format="svg")
            print(f"Mean firing rate plot saved as {plot_filename}")
        # plt.show()


def main():
    os.makedirs("plots", exist_ok=True)
    # === Global Parameters ===
    network_params = {
        # Modular Networks Experimental Setup (from Lecture 4 Topic 9)
        "NUMBER_OF_MODULES": 8,
        "EXCITATORY_PER_MODULE": 100,
        "INHIBITORY_NEURONS": 200,
        "CONNECTIONS_PER_MODULE": 1000,
        # Excitatory Izhikevich Neuron parameters (from Lecture 2 Topic 4)
        "excitatory_iz_neuron": {
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0,
            "a_r": 0.0,
            "b_r": 0.0,
            "c_r": 15.0,
            "d_r": -6.0,
        },
        # Inhibitory Izhikevich Neuron parameters (from Lecture 2 Topic 4)
        "inhibitory_iz_neuron": {
            "a": 0.02,
            "b": 0.25,
            "c": -65.0,
            "d": 2.0,
            "a_r": 0.08,
            "b_r": -0.05,
            "c_r": 0.0,
            "d_r": 0.0,
        },
    }
    simulation_time = 1000  # in ms

    # The p values specified in coursework
    P_VALUES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Dictionary to store the generated networks
    networks: dict[float, IzNetwork] = {}

    # === 2. Main Loop ===
    print("--- Starting network generation ---")

    for p in P_VALUES:
        print(f"--- Generating network for p = {p} ---")
        generator = ModularNetwork(p, network_params)

        generator.generate_modular_network()
        networks[p] = generator

        print(f"--- Finished p = {p} ---")
        # print(generator.run_simulation(simulation_time))
        spikes = generator.run_simulation(simulation_time)

        title = f"Connection matrix (p = {p})"
        generator.connectivity_matrix(
            excitatory_only=True,
            title=title,
            save_plot=True,
            plot_filename=f"plots/connection_matrix_p{p}.svg",
        )
        generator.raster_plot(
            spikes,
            excitatory_only=True,
            save_plot=True,
            plot_filename=f"plots/raster_plot_p{p}.svg",
        )
        generator.mean_firing_rate(
            spikes,
            simulation_time,
            window_size=50,
            step_size=20,
            include_inhibitory=False,
            save_plot=True,
            plot_filename=f"plots/mean_firing_rate_p{p}.svg",
        )

    print(f"--- All 6 networks generated ---")


if __name__ == "__main__":
    main()
