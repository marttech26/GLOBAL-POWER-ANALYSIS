# GLOBAL POWER ANALYSIS
![Power plants](power.jpg)

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
![Untitled](https://github.com/bulemi2/GLOBALPOWER-ANALYSIS/assets/133605850/8267e7ba-6372-46d3-88c9-fc1788cad08e)

The bar plot presents the top 70 regions with the highest total renewable electricity generation, showcasing China as the leader, followed by the United States, European Union, India, and Brazil. These leading regions collectively contribute a substantial portion of total renewable electricity generation. Additionally, the plot highlights disparities, revealing many regions with relatively low renewable electricity generation, particularly African countries and some Asian nations. This observation suggests untapped potential for renewable energy expansion in these regions. Overall, the bar plot emphasizes the significance of renewable energy contributions from leading regions while signaling opportunities for growth and development in regions with lower renewable electricity generation.
![Untitled](https://github.com/bulemi2/GLOBALPOWER-ANALYSIS/assets/133605850/42f33837-c945-40ff-9608-8810200afda4)

The plot depicts the global trend of renewable energy generation over time, showcasing a steady increase with fluctuations. Hydroelectric power dominates, followed by significant contributions from wind and solar power. Other renewables like bioenergy also play a role, albeit to a lesser extent. This data underscores global efforts to reduce emissions and expand renewable energy sources. However, challenges such as infrastructure, financing, and policy limitations remain.
![Untitled](https://github.com/bulemi2/GLOBALPOWER-ANALYSIS/assets/133605850/c85dba25-d810-47da-9006-8b4e5bbeb969)

## MODELING

The analysis compared the performance of three machine learning algorithms (Random Forest, Gradient Boosting, and Linear Regression) in predicting renewable energy production across different energy sources. 
 	Random Forest 	GradientBoosting
Wind	0.964	0.968
Hydro	0.980	0.975
Solar	0.851	0.895
Bioenergy	0.991	0.989
Total Renewable 	0.967	0.997
Fossil Fuels	0.971	0.972

We chose ARIMA model for its time series nature. 

Achieved MSE of 2590.25, RMSE of 50.894 overall, but notably lower for localized data: MSE ‘13.2’, RMSE ‘1.31243’.
Insights:
1. Linear Regression consistently performed well with low error metrics and high R-squared scores.
2. Random Forest and Gradient Boosting showed signs of overfitting, performing slightly worse on unseen data.
3. Cross-validation confirmed the robustness of the Linear Regression model.
4. The ARIMA model was applied to time series data, yielding reasonable fit statistics but with room for improvement.
5. Forecasting was effective for localized data but less accurate when aggregating data from multiple countries.
6. Detailed reports with visualizations and insights were generated for each energy source.

## Conclusion

The analysis highlights the importance of model selection and evaluation in predicting renewable energy production accurately. Linear Regression emerged as the most reliable model, emphasizing the need for further research to enhance forecasting accuracy, especially for heterogeneous datasets.
Hydropower Dominance: The analysis highlights hydropower's significant role in global renewable energy, evident from its strong correlation with total renewable electricity generation.

Diversification Potential: Positive correlations between wind and solar energy suggest opportunities for strategic portfolio diversification, aiding countries in balancing their renewable energy mix.

## Recommendations
Non-Renewable Dependency: Near-perfect correlation between non-renewable electricity and overall generation emphasizes continued reliance on non-renewable sources, urging a gradual shift towards sustainability.

Renewable Energy Growth: Wind, solar, and hydropower show promising growth trends, indicating a rising trajectory in global renewable energy adoption.

Strategic Investments: Optimize wind and solar energy investments using predictive modeling to target high-growth regions efficiently.

Hydropower Planning: Utilize modeling insights to strategically plan hydropower projects, enhancing efficiency and sustainability.

Diversification Strategies: Promote diversified renewable energy approaches, leveraging predictive models to identify optimal combinations of wind, solar, and hydropower.

Transition Policies: Align transition policies with predictive modeling, offering targeted incentives and regulations to support renewable energy growth areas.

Enhance Data Accessibility: Improve access to renewable energy data through user-friendly platforms, empowering stakeholders to make informed decisions.

Foster Collaboration: Facilitate collaboration among governments, organizations, and industry to advance renewable energy technologies and policies globally.

Embrace Innovative Solutions: Drive adoption of innovative solutions like smart grids and energy storage to address renewable energy challenges and promote scalability.

**Next Steps**

Refine and Validate Models:
- Continuously refine and validate the regression models, exploring additional variables and feature engineering techniques to improve accuracy.

Temporal Analysis:
- Assess renewable energy trends across various timeframes to ensure model robustness over time and capture temporal variations effectively.

Regional Sensitivity Analysis:
- Perform sensitivity analyses at regional levels to recognize unique challenges and characteristics, enabling tailored strategies to address specific regional needs.

