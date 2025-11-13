# padarthkhoj
#### Catalyst Activity Prediction using First-Principles and Graph Neural Networks and Interpretable Machine Learning

## padarthkhoj- OER/ORR
![OER/ORR](images/pi-pipeline.png)
![GUI](GUI_OER_ORR.jpg)

The graphical interface for OER and ORR operates through a streamlined sequence of steps. To begin, each material system undergoes two separate DFT computations: one dedicated to relaxing the atomic geometry and another to generate the density of states (DOS). After these simulations, the DOSCAR output is imported into the GUI. Internally, the interface applies the split_doscar routine to separate and compute the projected density of states (PDOS) for individual atoms. From the PDOS profiles, two key electronic descriptors are derived. These electronic descriptors estimates the energy descriptors and this energy descriptor serve as input to a predictive model that estimates the overpotential associated with OER and ORR activity in sp²-hybridized systems.


## padarthkhoj- CO2ER_1
![Part I Workflow](images/CO2ER.png)




## padarthkhoj- CO/CO2ER
![Part I Workflow](images/CO_CO2ER.jpg)




## padarthkhoj- NRR
![Part I Workflow](images/NRR.jpg)
