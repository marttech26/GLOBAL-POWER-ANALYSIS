# GLOBAL ENERGY ANALYSIS

## Introduction
Renewable energy, encompassing diverse sources such as solar, wind, hydro, and geothermal power, offers a pathway towards decarbonizing our energy systems and mitigating the adverse effects of fossil fuel consumption. Recognizing the critical importance of this transition, this project endeavors to delve into the realm of renewable energy. Our goal is to analyze global data on renewable energy technologies, shedding light on their current contributions to the energy mix and the pace of their evolution. In doing so, we seek to provide valuable insights to stakeholders, shaping the discourse on sustainable energy and enriching our collective knowledge of renewable energy infrastructure worldwide.

## Imported Libraries
- pandas
- matplotlib.pyplot
- seaborn
- numpy
- pickle
- datetime
- math

## Business Understanding
In the current era marked by energy transitions and growing sustainability imperatives, the Global Renewable Energy Generation and Capacity Analysis project endeavors to decode the intricacies of the global renewable energy landscape. Through an in-depth exploration of various renewable energy sources, their capacities, and their contributions to renewable energy generation, the project aims to offer a comprehensive overview of the global renewable energy ecosystem. These insights will equip stakeholders, policymakers, and investors with valuable knowledge about the factors influencing the renewable energy industry, enabling informed decision-making in this rapidly evolving and crucial sector.

This Global Renewable Energy Analysis is set to benefit the following stakeholders:
- Power generation companies: Strategic planning
- Environmental Agencies: Providing environmental impact assessment
- Researchers and Academia: Contributing to advancements in energy studies
- Local communities: Informed community engagement
- Technology providers: Market identification and growth opportunities
- Government and Regulatory Bodies: Informed decision making
- Energy companies and investors looking for opportunities in the renewable energy market.

## Problem Statement
Since the Industrial Revolution, fossil fuels have dominated the global energy mix, leading to significant greenhouse gas emissions and health issues. To combat these challenges, there's a pressing need to transition to low-carbon energy sources like nuclear and renewables. Renewable energy, in particular, is crucial for reducing CO2 emissions and air pollution. Despite the availability of existing analysis on global renewable energy, there remains a critical gap in translating this wealth of information into actionable insights for stakeholders. Therefore, there is a pressing need for a tailored approach to renewable energy analysis that addresses the specific needs and challenges faced by stakeholders at regional and local levels.

This project aims to fill this gap by developing a framework for localized renewable energy analysis that considers the unique characteristics and dynamics of each region.

## Main Objective
To construct a predictive model utilizing worldwide energy data to precisely anticipate the generation of various energies from other energy sources.

### Specific Objectives
- To assess the role each renewable energy source plays in the overall energy blend.
- To identify opportunities to enhance the portion of renewable energy in global electricity production, optimizing the shift towards sustainable sources.
- To examine how renewable energy is distributed among different power regions.
- To identify which renewable source shows the most significant growth.
- To identify disparities in renewable energy adoption between developed and developing countries and explore the underlying factors contributing to these disparities.

## Data Understanding
We obtained our dataset from 'Our World in Data,' a comprehensive source of global statistics covering various aspects of energy. By merging two distinct CSV files from 'Our World in Data,' we created a dataset comprising 7165 rows and 9 columns. This dataset encompasses annual energy data for major electricity sources from 2000 to 2022.

The merged dataset contains the following columns with their descriptions:
- Entity (text): This column represents the geographical entity or region for which the energy data is recorded.
- Year (number): This column represents the year in which the energy data was recorded or measured.
- Electricity from wind - TWhr (number): This column represents the amount of electricity generated from wind energy in terawatt-hours (TWh). It indicates the contribution of wind energy to the total electricity generation in the specified entity and year.
- Electricity from hydro - TWh (number): This column represents the amount of electricity generated from wind energy in terawatt-hours (TWh). It indicates the contribution of hydroelectric sources to the total electricity generation in the specified entity and year.
- Electricity from solar - TWh (number): This column represents the amount of electricity generated from wind energy in terawatt-hours (TWh). It indicates the contribution of solar energy sources to the total electricity generation in the specified entity and year.
- Other renewables including bioenergy - TWh (number): This column represents the combined amount of electricity generated from other renewable sources, such as biomass, geothermal, and tidal energy, excluding wind, hydro, and solar. It's also measured in terawatt-hours (TWh).
- Electricity from Non-Renewables - TWh (number): This column represents the amount of electricity generated from non-renewable sources, such as fossil fuels (coal, oil, natural gas) and nuclear power, in terawatt-hours (TWh).
- Total Renewable Electricity - TWh (number): This column represents the total amount of electricity generated from renewable sources, including wind, hydro, solar, and other renewables, in terawatt-hours (TWh).
- Electricity generation - TWh (number): This column represents the total electricity generated from all sources, both renewable and non-renewable, in terawatt-hours (TWh).

These columns provide a comprehensive overview of the electricity generation landscape, detailing the contributions of various renewable and non-renewable sources over time.

## Data Preparation
Data preparation involved merging several data files into a single DataFrame using pandas. Extensive data cleaning was performed, including missing value treatment, duplicate removal, and feature engineering.

## Exploratory Data Analysis
Both univariate and bivariate analyses were performed:
The line plot illustrates the comparison between total renewable electricity generation and electricity generation from non-renewable sources over the years. It demonstrates that renewable energy sources, including wind, hydro, solar, and other renewables, consistently contribute more to electricity generation compared to non-renewable sources such as coal, natural gas, and nuclear energy. In essence, renewable energy sources emerge as the primary contributors to electricity generation in the dataset, with non-renewable sources playing a secondary role. This perspective highlights the dominance of renewable energy in the dataset's electricity generation dynamics over time.


The bar plot presents the top 70 regions with the highest total renewable electricity generation, showcasing China as the leader, followed by the United States, European Union, India, and Brazil. These leading regions collectively contribute a substantial portion of total renewable electricity generation. Additionally, the plot highlights disparities, revealing many regions with relatively low renewable electricity generation, particularly African countries and some Asian nations. This observation suggests untapped potential for renewable energy expansion in these regions. Overall, the bar plot emphasizes the significance of renewable energy contributions from leading regions while signaling opportunities for growth and development in regions with lower renewable electricity generation.
![Untitled](https://github.com/bulemi2/GLOBALPOWER-ANALYSIS/assets/133605850/01569c26-c837-4daf-970e-cdafa952f043)




