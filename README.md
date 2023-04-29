# Project-Bev-Phev
				
 
Introduction
Comparison between battery and plug in hybrid Electric vehicles can be essential nowadays. The purpose of this project is to help determine the best and alternative options of vehicles make and model for someone who is interested in purchasing a battery electric vehicle (BEV) or plug-in hybrid electric vehicle (PHEV).  Our analysis is based on battery capacity, battery recharge time maintenance cost, battery range, price of the vehicle and incentives. The government of Canada gives incentives to encourage the use of BEV and PHEV, to help reduce Carbon dioxide emissions and smog to the atmosphere. Not all the cars are getting incentives as only cars that have a price lower than $55k are eligible.
Objectives 
To help customers buy the best suited BEV or PHEV vehicle based on their needs.
Problem Solving Process
•	Exploring the Mess: Increasing level of carbon dioxide in the atmosphere becoming the cause of depletion of ozone layer. To save the environment from global warning as it has adverse effect on our health and climate. To decrease the use of gas vehicle and encourage people to buy more EV by providing them incentives. The goal is becoming carbon free while growing GDP.
•	Searching for the information: Significantly limit or stop the use of equipment that leads to the emission of Carbon dioxide emissions and other gases causing a depletion in the ozone layer. This includes reducing the number of fuel vehicles, refineries, and other industries and equipment that release these gases to the atmosphere. Discourage the use of fuel vehicles by giving incentives to BEV and PHEV. Produce efficient and less costly BEV and PHEV which can travel a significant amount of distance on electricity when their batteries are fully charged and limit the production of fuel vehicles Increasing carbon tax
•	Identifying the problem: The consequences are extreme weather that may significantly affect human health and the ecosystem. Global warming will lead to the melting of polar ice shields and consequently a rise in sea level, affecting coastal habitats, wetland flooding, soil contamination with sea salt, etc
•	Searching for solution: Searching for the different set of data available on all the open site to find the battery capacity battery recharge different and different set of data that helps to make decision for the customers to find the best suited product.
•	Evaluating solution: It should have positive impact on the climate change when the level of CO2 emission decreased in the atmosphere and change the fuel used in the transport system. Plantation of trees, change in water & agriculture systems. It will stabilization of ecosystems and innovation of new Technology (example. more efficient batteries)
•	Implementing Solution: Different people have different opinions in terms of switching to electric vehicles and limited number of charging stations that effects the sales of electric vehicles. Difficult in accepting new technology.

Business questions
A comparison between BEV and PHEV is possible by analyzing the following business questions.
•	Which vehicle model/make will give more incentives? 
•	Do people prefer battery or plug-in hybrid electric vehicles? 
•	Does the vehicle price and battery capacity influence decision-making? 
•	Which vehicles have high/low charging time and high/low maintenance cost?  
•	Which vehicles have better range?
Can we make decision about buying the best suited electric/hybrid vehicle for potential customers?
Datasets and source
Three datasets were gotten from the government of Canada open government portal.
List of eligible vehicles under the iZEV Program (canada.ca)
Fuel consumption ratings - Battery-electric vehicles 2012-2021 (2021-08-12) - Open Government Portal (canada.ca)
Fuel consumption ratings - Plug-in hybrid electric vehicles 2012-2021 (2021-08-12) - Open Government Portal (canada.ca)
Data Cleansing
•	Deleted all vehicle’s data before 2019 from the 2008-2022 BEV and PHEV datasets.
•	Created a dataset with all the Electric and plug in hybrid models from incentive dataset, a dataset for only BEV and another for only PHEV.
•	Created a dataset for Battery capacity and price by using different search engines (internet).
•	Creating key column in all the datasets to establish relationships.
•	Used multiple imputations to fill up missing values.
•	Dropped unnecessary data columns from the dataset.
•	Alberta's Enmax current rate was used to calculate the fuel cost to travel 100 km.
•	Calculated cost of fuel to travel 100 km based on the range of batteries and range of gas by using the cost of current electricity and current
prices of Regular and Premium gasoline.
Descriptive Statistics
Descriptive statistics was used to provide basic information about the variables in the dataset and to highlight potential relationships. There were 
4 different amounts offered for incentives: $1250, $2500, $3750, and $5000 for a lease of 12, 24, 36, and 48 months respectively. The datasets contain 
a total of 234 records of vehicles from different Makes, Models, and Year. Three and fourteen were models for the years 2018 and 2022 respectively while 
the remaining 217 records came from models in the years 2019, 2020, and 2021. 
 
 Pricing:The prices of plugin hybrid vehicles are range from 24K and the maximum price of Porsche made is around 2k. Most of vehicles range from 34K to84k 
 for PHeV'whereas for BEV it ranges from 42K to 78K.The average price of plug-in vehicles are 64K whereas for BEV is 67K and More number of plug in hybrid vehicles 
are available in the market as compared to BEV.

Battery Range and Cost of Fuel: Battery Range for gas vehicles depend upon the range of battery and range of gas so it is always higher than the range of BEV. 
The cost of fuel for BEV varies very little as the prices of electric is low whereas for the gas vehicles it depends on the size of battery if the size of battery 
is big then the cost of fuel is less but if the size of battery is small so vehicle will use more of the gas so prices goes up depending upon the regular and premium
gasoline.

  
Battery capacity and recharge time:As show in the bar plots, the battery capacity and recharge time mostly depend on each other as if the battery capacity is large,
then it will probably take more time to recharge it. The line plot shows that the recharge time of PHEV was in the range 1.5 to 4 hours while that of BEV was between
7 and 13 hours. This is an indication that BEVs had larger battery capacities as compared to PHEVs. Other important variables not shown in this exploratory but used in
the analysis were battery range, vehicle price, fuel cost, CO2 emissions and smog ratings. 

Statistical analysis: We have conducted a univariate and multivariate regression of the price for two purposes. We had some missing values in our dataset. So, to fill
the missing value we did multiple imputation by doing regression analysis of several subsets of the dataset. We have also tried to understand any correlation with 
other variables such as total range of the vehicle in full charge/full tank, size of the battery, recharge time, vehicle class, make, model, year, etc. We have found 
an R-squared value of 0.466 that indicates that the variables can predict 46.6% variance of the price. Seating capacity and total range were negatively correlated, and all other variables were positively correlated to some extent. 
Scenarios / Answers to business questions Based on our analysis we have deployed a decision-making tool to help choose an electric/hybrid vehicle based on an individual’s personal preference, budget, need, and constraints. 
 
Using above dashboard, we can shortlist the best BEV/PHEV vehicle(s) for the client. The following scenario can be used as an illustration.
Scenario: An elderly couple want to buy an electric two-seater. Their budget is below 40k. They are looking for just a car to drive in within the city. They want a low maintenance and if incentives are available that would be great. 
Putting these inputs in our solution we found smart EQ vehicles from 2019-2020 model that the couple didn't like. So, we went to 4-seater and found a MINI Cooper within these requirements.
In addition to this, using our regression model we can give an estimate of the vehicle price, maintenance cost, and available incentive amount for a car coming in next years, if the clients are interested in buying that. 
Conclusion 	
With the rise in electric vehicles, people will have to make decisions in the near future as to which model to purchase. Battery and plug-in hybrid electric vehicles can improve fuel economy, lower fuel cost, and reduce emissions. This project was to analyze BEV and PHEV vehicles based on variables such as government incentives, vehicle’s price, battery capacity, range and recharge time, fuel cost and car make, model and year to help provide options to those interested in purchasing these vehicles. Using analysis, we could provide the best model as well as alternatives based on a client’s preference.
General speaking, BEV had more incentives, bigger battery capacity, longer range, more charging time, and lower fuel/electric cost as compared to PHEV. The price of the vehicle mostly depends on the make and model with BEV having a slightly higher price than PHEV. All these factors will influence people’s preferences and decisions. It will be a give and take for people to determine the type of BEV or PHEV that they want.
Nevertheless, it is difficult to say where the market of these vehicles is going to look like. Thus, think before you decide to flip the coin. We can either watch it happen or be part of it. 

