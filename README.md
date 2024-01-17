# Demo: How to Apply FlashSyn to a New Contract

In this demonstration, we illustrate the process of applying FlashSyn to a new smart contract, using "Puppet" from Damn Vulnerable DeFi as a case study.

The primary repository for the artifact is available at [FlashSyn-Artifact/FlashSyn-Artifact-ICSE24](https://github.com/FlashSyn-Artifact/FlashSyn-Artifact-ICSE24). The artifact repository serves as a key resource for replicating our evaluation results as presented in our ICSE 2024 paper.

This repository, on the other hand, is specifically for a demo how to apply FlashSyn to a new contract. To streamline the demo experience, we have omitted several files that are not essential, retaining only the core components of FlashSyn. Our goal is to showcase the application of FlashSyn on the new contract "Puppet" and provide insights into the utilization and adaptation of FlashSyn for building custom tools.

## Installation

FlashSyn requires the following dependencies:

1. **Foundry**: Detailed instructions and resources can be found at [Foundry GitHub Repository](https://github.com/foundry-rs/foundry).
2. **Python 3.7 or above**: Ensure you have a compatible Python version.
3. **Python Packages**: Install the required packages using `pip install -r requirements.txt`.

For comprehensive installation details, refer to the [Dockerfile](https://github.com/FlashSyn-Artifact/FlashSyn-Artifact-ICSE24/blob/main/Dockerfile) in the artifact repository.



## Step 1: Setting Up a Testing Environment in Foundry

The `src/foundryModule/` directory is designated for establishing the Foundry testing environment. We assume familiarity with Foundry; if needed, please consult the [Foundry Repository](https://github.com/foundry-rs/foundry) for more information.

FlashSyn executes all Foundry commands in the `src/foundryModule/` directory by default. We have set up a testing environment for Puppet similar to all other Foundry projects.  

The Puppet Foundry templates are sourced from [this repository](https://github.com/nicolasgarcia214/damn-vulnerable-defi-foundry), based on the original challenge found at [Damn Vulnerable DeFi](https://www.damnvulnerabledefi.xyz/challenges/puppet/). 


## Step 2: Testing the Executability of the Actions

Post-environment setup, it's important to test the executability of the actions of interest.

The file `src/foundryModule/src/test/Levels/puppet/Puppet1.t.sol` provides an example of testing actions. Each action is accompanied by a unit test to verify its executability. Parameters are selected for ease of testing, without considering potential attack vectors at this stage.

Execute `forge test --match-contract PuppetStep1` to ensure all actions are executable.

## Step 3: Annotating Action Candidates

Once the actions' executability is confirmed, the next step involves annotating action candidates.

For a detailed guide on this process, refer to Appendix C in the extended version of our paper available at [arXiv](https://arxiv.org/pdf/2206.10708.pdf).

The file `src/foundryModule/src/test/Levels/puppet/Puppet2.t.sol` demonstrates the annotated actions.

Run `forge test --match-contract PuppetStep2` to test these actions and their annotations. FlashSyn uses these annotations to collect data points and assess profitability, crucial for the upcoming steps.

## Step 4: Writing a Python Script for Action Specifications

With annotated actions, the next phase involves scripting the action specifications in Python.

`src/Actions/Puppet.py` exemplifies this for the Puppet scenario. It breaks down the annotated Foundry script from Step 2 into segments and defines specifications for each action.

Although this Python file may look tedious at the first glance, the structure of the Python file is patterned and contains many repetitive elements.  

Please take a look at other Python files in the artifact repo [src/Actions/](https://github.com/FlashSyn-Artifact/FlashSyn-Artifact-ICSE24/blob/main/src/Actions/) directory for more examples.


## Step 5: Running FlashSyn - Synthesizing Attack Vectors

The final step is to execute FlashSyn for synthesizing attack vectors.

Run `python3 src/Actions/Puppet.py` to initiate this process. FlashSyn will conclude by presenting the synthesized attack vectors and the highest global profit identified. A positive profit from any attack vector indicates a successful identification of a Flash Loan attack vector by FlashSyn.

---

**Note**: This artifact is a prototype. Several steps in the process have the potential for automation and optimization.
