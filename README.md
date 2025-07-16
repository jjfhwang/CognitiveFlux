# CognitiveFlux: Neuro-Symbolic Reasoning Engine

A powerful Python-based neuro-symbolic engine for building robust and explainable AI agents by seamlessly integrating differentiable neural modules with logic programming.

## Detailed Description

CognitiveFlux is a novel framework designed to bridge the gap between the strengths of neural networks and symbolic reasoning. While neural networks excel at pattern recognition and learning from data, they often lack explainability and struggle with logical inference. Conversely, logic programming provides a clear and interpretable way to represent knowledge and perform deductive reasoning, but it can be brittle and difficult to integrate with real-world, noisy data. CognitiveFlux addresses these limitations by providing a unified platform where neural modules and symbolic rules can interact and complement each other.

The engine allows developers to construct hybrid AI systems that leverage the learning capabilities of neural networks for tasks like perception, prediction, and classification, while simultaneously utilizing symbolic rules for reasoning, planning, and decision-making. This integrated approach results in AI agents that are not only accurate but also capable of providing justifications for their actions, making them more transparent and trustworthy. CognitiveFlux aims to facilitate the development of AI systems that are more resilient to adversarial attacks, capable of handling out-of-distribution data, and easier to debug and maintain compared to purely neural approaches.

CognitiveFlux facilitates the creation of complex reasoning systems by providing a flexible architecture for defining and connecting neural modules and logic programs. The system supports various neural network architectures, including feedforward networks, convolutional networks, and recurrent networks, as well as different logic programming paradigms, such as Prolog and Datalog. Through a unified interface, users can define the interactions between these components, enabling the system to reason about the outputs of neural networks using symbolic rules and vice versa. The framework supports backpropagation through the entire system, allowing for end-to-end training of both neural and symbolic components, further enhancing the integration and performance of the hybrid system.

## Key Features

*   **Differentiable Logic Programming Integration:** Enables seamless integration of differentiable neural networks with symbolic logic programming paradigms, allowing for gradient-based training of the entire system. This feature uses custom gradient implementations for logical operators to backpropagate through symbolic rules.
*   **Modular Architecture:** Offers a highly modular design, allowing developers to easily incorporate custom neural modules (e.g., image classifiers, language models) and logic programs written in various dialects (e.g., Prolog, Datalog). Modules can be added or removed without affecting other system components.
*   **Knowledge Representation with Hybrid Structures:** Supports the representation of knowledge using a combination of neural embeddings and symbolic facts. This allows the system to reason about both continuous and discrete data, enabling more comprehensive knowledge representation.
*   **Explainable Reasoning:** Provides mechanisms for tracing the reasoning process, allowing users to understand how the system arrives at its conclusions. This is achieved through rule-based explanations derived from the logic program's execution trace.
*   **Rule-Based Inference with Neural Guidance:** Incorporates neural network outputs to guide the inference process in logic programming. Neural modules can provide probabilities or confidence scores that influence the selection of rules to apply during reasoning.
*   **End-to-End Training:** Supports end-to-end training of the entire neuro-symbolic system, optimizing both neural and symbolic components jointly. This is achieved using custom loss functions that combine performance metrics from both neural and symbolic outputs.
*   **Support for Multiple Logic Programming Dialects:** The system is designed to be adaptable and supports multiple logic programming dialects, including Prolog and Datalog, allowing developers to choose the language that best suits their needs.

## Technology Stack

*   **Python:** The core programming language used for building the framework.
*   **PyTorch:** A powerful deep learning framework used for implementing and training neural modules. PyTorch's dynamic computation graph is essential for supporting the differentiable logic programming aspects.
*   **Numpy:** A fundamental package for scientific computing in Python, used for numerical operations and array manipulation.
*   **SWI-Prolog (via pyswip):** A Prolog implementation used for logic programming within the system. pyswip acts as a bridge between Python and SWI-Prolog.
*   **(Optional) TensorFlow:** Alternative deep learning framework that can be utilized for the neural modules.

## Installation

1.  **Clone the repository:**
    git clone https://github.com/jjfhwang/CognitiveFlux.git
    cd CognitiveFlux

2.  **Create a virtual environment (recommended):**
    python3 -m venv venv
    source venv/bin/activate

3.  **Install the required dependencies:**
    pip install -r requirements.txt

    Ensure you have SWI-Prolog installed on your system and that the `pyswip` library can access it. Installation instructions for SWI-Prolog can be found on the official SWI-Prolog website. On Debian/Ubuntu systems, you can use: `sudo apt-get install swi-prolog`.

4.  **(Optional) Install TensorFlow:**
    If you intend to use TensorFlow-based neural modules, install it with `pip install tensorflow`.

## Configuration

CognitiveFlux relies on certain environment variables for proper configuration. Create a `.env` file in the root directory of the project to set these variables.

*   `PROLOG_PATH`: The path to the SWI-Prolog executable. If SWI-Prolog is in your system's PATH, this is not required. Example: `PROLOG_PATH=/usr/bin/swipl`.

You can load the environment variables into your Python script using the `dotenv` library.

## Usage

Example: Integrating a simple neural classifier with a Prolog knowledge base.

First, define a simple neural network in PyTorch (or TensorFlow):

class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

Then, define a Prolog knowledge base:

/* facts */
animal(dog).
animal(cat).

/* rules */
likely_pet(X) :- animal(X), classifier_confidence(X, Confidence), Confidence > 0.7.

Now, integrate the neural classifier's output with the Prolog rules:

from CognitiveFlux.core import CognitiveEngine
from pyswip import Prolog

# Initialize the Cognitive Engine
engine = CognitiveEngine()

# Load the Prolog knowledge base
engine.load_prolog_rules("knowledge_base.pl")

# Define a function to get the classifier's confidence
def get_classifier_confidence(animal):
    # Simulate a neural network classification
    if animal == "dog":
        return 0.8
    elif animal == "cat":
        return 0.9
    else:
        return 0.2

# Add the classifier confidence as an external predicate
engine.register_external_predicate("classifier_confidence", get_classifier_confidence)

# Query the Prolog engine
results = engine.query("likely_pet(X).")
for result in results:
    print(result["X"])

This example demonstrates how a neural network's classification confidence can be used within a Prolog rule to infer whether an animal is likely to be a pet. The `CognitiveEngine` manages the interaction between the neural network (simulated here) and the Prolog environment. More complex examples involving backpropagation through differentiable logic programs can be found in the `examples/` directory.

## Contributing

We welcome contributions to CognitiveFlux! Please follow these guidelines:

*   Fork the repository.
*   Create a new branch for your feature or bug fix.
*   Write clear and concise commit messages.
*   Submit a pull request with a detailed description of your changes.
*   Ensure your code adheres to PEP 8 style guidelines.
*   Include unit tests for any new functionality.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/jjfhwang/CognitiveFlux/blob/main/LICENSE) file for details.

## Acknowledgements

We would like to acknowledge the contributions of the open-source community to the PyTorch, TensorFlow, and SWI-Prolog projects, which form the foundation of CognitiveFlux.