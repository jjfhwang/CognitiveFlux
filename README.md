# CognitiveFlux: Harnessing Neural Dynamics for Efficient Computation

CognitiveFlux is a Python library designed to emulate and leverage the principles of neural dynamics for enhanced computational efficiency and adaptive problem-solving. It draws inspiration from the brain's capacity to rapidly adjust its internal state based on incoming information, allowing for dynamic and context-aware processing. This library provides a flexible framework for simulating neural networks and applying related algorithms to diverse domains, from time series analysis and pattern recognition to optimization and control. It emphasizes modularity and extensibility, enabling researchers and developers to experiment with different network architectures, learning rules, and applications.

At its core, CognitiveFlux offers a set of tools for constructing and simulating spiking neural networks (SNNs) and rate-based neural networks. It incorporates various neuron models, including Leaky Integrate-and-Fire (LIF) and Hodgkin-Huxley models, and supports different synaptic plasticity mechanisms, such as Spike-Timing-Dependent Plasticity (STDP) and Hebbian learning. The library also provides utilities for data preprocessing, visualization, and performance evaluation. By abstracting away the complexities of low-level neural simulations, CognitiveFlux allows users to focus on the high-level design and application of neural-inspired algorithms. Its goal is to bridge the gap between neuroscience research and practical computational solutions.

CognitiveFlux is not just a simulation tool; it is designed to facilitate the development of novel algorithms inspired by neural dynamics. By providing a platform for experimenting with different network architectures and learning rules, the library encourages the exploration of new approaches to solving complex problems. Its adaptive nature allows it to adjust its internal parameters based on the characteristics of the input data, making it particularly well-suited for applications where the data distribution is non-stationary or the environment is constantly changing. Furthermore, the modular design of the library enables users to easily integrate it with other Python libraries, such as NumPy, SciPy, and TensorFlow, for a wide range of applications.

Key Features:

*   **Spiking Neural Network (SNN) Simulation:** Provides a flexible framework for simulating SNNs with customizable neuron models (LIF, Hodgkin-Huxley) and synaptic plasticity rules (STDP, Hebbian). The core simulation engine is optimized for performance, leveraging vectorized operations and parallel processing capabilities.

*   **Rate-Based Neural Network Simulation:** Supports the simulation of rate-based neural networks, where neuron activity is represented by firing rates rather than individual spikes. This allows for efficient computation in scenarios where the precise timing of spikes is not critical. The rate-based models are implemented using differential equations solved with numerical integration methods from SciPy.

*   **Synaptic Plasticity Mechanisms:** Implements various synaptic plasticity mechanisms, including STDP, Hebbian learning, and homeostatic plasticity. These mechanisms allow the network to learn and adapt to changing input patterns. Custom plasticity rules can be defined through a flexible callback interface.

*   **Data Preprocessing Utilities:** Offers a suite of data preprocessing tools specifically designed for neural network applications, including normalization, standardization, and feature extraction. These tools ensure that the input data is in a suitable format for training and simulation.

*   **Visualization Tools:** Provides visualization tools for analyzing network activity, including spike raster plots, firing rate histograms, and synaptic weight distributions. These tools aid in understanding the behavior of the network and diagnosing potential issues.

*   **Modular Architecture:** Designed with a modular architecture that allows for easy extension and customization. New neuron models, synaptic plasticity rules, and simulation algorithms can be added without modifying the core library code.

*   **Integration with Scientific Python Ecosystem:** Seamlessly integrates with other popular Python libraries, such as NumPy, SciPy, and TensorFlow, allowing users to leverage a wide range of tools for data analysis, optimization, and machine learning.

Technology Stack:

*   **Python 3.7+:** The primary programming language used for the library.

*   **NumPy:** Used for efficient numerical computations and array manipulation. Its vectorized operations are crucial for the performance of the simulation engine.

*   **SciPy:** Provides scientific computing tools, including numerical integration methods and optimization algorithms. The library uses SciPy for solving differential equations in neuron models and for parameter optimization.

*   **Matplotlib:** Used for creating visualizations of network activity and simulation results. Its plotting capabilities are essential for analyzing the behavior of the network.

*   **Numba (Optional):** An optional dependency that can be used to further accelerate the simulation engine by compiling Python code to machine code at runtime.

Installation:

1.  Ensure that you have Python 3.7 or later installed on your system.
2.  Create a virtual environment (recommended):
    python3 -m venv cognitiveflux_env
    source cognitiveflux_env/bin/activate
3.  Clone the CognitiveFlux repository:
    git clone https://github.com/jjfhwang/CognitiveFlux.git
4.  Navigate to the CognitiveFlux directory:
    cd CognitiveFlux
5.  Install the required dependencies:
    pip install -r requirements.txt
    (Optional) Install Numba for performance improvements:
    pip install numba

Configuration:

CognitiveFlux utilizes environment variables for certain configuration options. These variables can be set in your shell environment or in a `.env` file in the project root directory.

*   `COGNITIVEFLUX_LOG_LEVEL`: Sets the logging level for the library. Possible values are "DEBUG", "INFO", "WARNING", "ERROR", and "CRITICAL". Default is "INFO".
*   `COGNITIVEFLUX_SIMULATION_THREADS`: Specifies the number of threads to use for parallel simulations. Default is the number of available CPU cores.

To set an environment variable in Linux/macOS:
export COGNITIVEFLUX_LOG_LEVEL=DEBUG

To set an environment variable in Windows:
set COGNITIVEFLUX_LOG_LEVEL=DEBUG

Usage:

Example: Simulating a simple LIF neuron

import numpy as np
from cognitiveflux.neurons import LIFNeuron

# Create a LIF neuron
neuron = LIFNeuron(tau_m=20.0, v_rest=-70.0, v_thresh=-55.0)

# Define input current
time = np.arange(0, 100, 0.1)
input_current = np.zeros_like(time)
input_current[20:80] = 1.0  # Apply current pulse

# Simulate the neuron
voltage = neuron.simulate(input_current, time)

# Print the membrane potential at the end of the simulation
print("Final membrane potential:", voltage[-1])

Detailed API documentation is available in the `docs` directory of the repository. This includes descriptions of all classes, functions, and modules within the CognitiveFlux library, along with examples of their usage.

Contributing:

We welcome contributions to CognitiveFlux! Please follow these guidelines:

1.  Fork the repository and create a new branch for your feature or bug fix.
2.  Write clear and concise code with comprehensive documentation.
3.  Follow the existing code style and conventions.
4.  Write unit tests to ensure that your code is working correctly.
5.  Submit a pull request with a detailed description of your changes.

License:

This project is licensed under the MIT License. See the [LICENSE](https://github.com/jjfhwang/CognitiveFlux/blob/main/LICENSE) file for details.

Acknowledgements:

We would like to acknowledge the contributions of the open-source community and the researchers whose work inspired the development of CognitiveFlux. Special thanks to the developers of NumPy, SciPy, and Matplotlib for providing the foundational tools for scientific computing in Python.