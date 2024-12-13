# Data Visualizer and Correlator

Welcome to our data characterization library and interface! Regardless of the field of work you are involved in, collecting and processing data is ubiquitous and must be handled with care. Gaussian distributions are very common and useful when dealing with random variables, but the data won't always follow this clean normal distribution. Data can follow a wide range of distribution shapes, so we created this tool to help scientists have a better understanding of their data sets. At the moment, this tool can handle and analyze gaussian, gamma, inverse-gamma, uniform, and lognormal distributions. Included is a script to randomly generate sample data for each of the above distribution models, outputting to CSVs in the data folder. 


## Data Generator:

The data folder is populated by data_maker.py, which currently makes a 1000 point sample for every single distribution type. Each of these output files is ready to be used with the GUI, and will have the highest p-value with their respective shape when compared.

## Correlation Calculator:

The correlation_generator.py script accepts an input data vector in order to compare against "standard" data for each of the distribution shapes in question. A Kolmogorov-Smirnov test is then run, comparing the data against an ideal curve with defined parameters. The output p-value is essentially the likelihood that the two data sets could be drawn from the same population. Since the models are idealized and won't match perfectly, this tool is more useful in showing the researcher which models most align with their input data. P-values are generated to compare data against each distribution shape individually.

## GUI Overview:

The GUI starts by accepting an input data file, and shows a preview of the data to make sure it imported correctly. The GUI will then display an estimated distribution density curve of the data sorted by value. The estimation makes a continuous curve from discrete data. A dropdown is above the density estimation plot, giving the user the option to overlay a standard curve with their data. This gives a direct visual comparison of distribution shapes, supplementing the correlation calculator's output. The second tab allows the user to generate a custom random sample with adjustable shape parameters, sample size, and mean. Any of the above distribution types are options for generation.

## Usage:  

Data generation (in terminal):  
python data_maker.py  
  
GUI activation:  
streamlit run gui.py  
If you are not redirected automatically, ctrl+click the local url link output to the terminal  
stop the program in your terminal with ctrl+c  