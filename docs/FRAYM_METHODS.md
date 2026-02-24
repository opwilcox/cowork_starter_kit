# Fraym Methodology Reference

This document provides detailed information about Fraym's data collection, processing, and modeling methodology.

## About Fraym

Fraym provides high-resolution geospatial data and analytics focused on population characteristics, behaviors, and attitudes globally. Fraym data scientists combine survey data, satellite imagery, large-scale geospatial datasets, and machine learning to generate insights at fine geographic resolutions.

## What is Fraym Data?

Fraym has built an artificial intelligence / machine learning (AI/ML) technology that weaves together high-quality household survey data with satellite imagery and other demographic and socioeconomic data covariates to create localized population information. Fraym's surveys include questions on:

- Demographics
- Socioeconomics
- Norms
- Behaviors
- Motivation and ability
- Media Use
- Influencers
- And other indicators

Fraym's AI/ML technology uses geotagged survey responses, along with satellite imagery layers, to predict the percentage of respondents who exhibit the given characteristic within each one km² grid cell (e.g., the percentage of the population who completed high school; the percentage of the population who use contraceptives). Fraym has provided a Codebook detailing specific definitions and the related survey question for each indicator provided.

## Data Inputs

### Survey Data (Primary Input)

The first data input is primary data from scientifically sampled, geo-tagged household surveys. Fraym has collected or accessed, cleansed, and harmonized more than 1,000 surveys from around the world. Key indications of a high-quality household survey include implementing organization(s), sample design, sample size, and response rates. After data collection, post-hoc sampling weights are created to account for any oversampling and ensure representativeness.

### Satellite Imagery and Covariates (Secondary Input)

The second major data input is satellite imagery and related derived data products, including:
- Earth observation (EO) data
- Gridded population information (e.g., human settlement mapping)
- Proximity to physical locations (e.g., health clinics, ports, roads, etc.)
- Proximity to significant events (e.g., political protests, labor strikes, extremist attacks)
- Biophysical surfaces like soil characteristics
- Demographic and socioeconomic covariates derived from national population and housing censuses or other large official household surveys

As with the survey data, Fraym data scientists ensure that the software only uses high-quality imagery and derivative inputs.

## Survey Data Collection Process

Fraym applies industry best practices to ensure that all geotagged survey training data for our AI/ML models is of the highest standard. Key indications of a high-quality household survey include implementing organization(s), sample design, sample size, and response rates. Fraym regularly works with vendors to carry out large-scale household surveys. This quality assurance and quality control (QA/QC) process entails five key steps:

### 1. Survey Vendor Selection

Fraym conducts extensive and ongoing reviews of major survey vendors globally. Our selection criteria include a range of factors, such as ESOMAR certification, QA/QC protocols and controls, size and scale, geographic coverage (global, regional, country-specific), and other considerations. We will only work with the highest quality and most reputable survey vendors.

### 2. Survey Panel Composition and Sample Representativeness

Fraym carefully considers sample representativeness measures in conjunction with the survey vendor selection process. For existing respondent panels, we will closely examine recruitment and retention methods. This includes a well diversified set of online and offline recruitment activities to ensure appropriate representation of difficult to reach demographic groups. Moreover, we stress test the vendors' ability to meet Fraym's demanding sampling quotas across demographic, socioeconomic and geographic dimensions. Outside of existing respondent panels, Fraym will only work with vendors who meet best practice standards for probabilistic sampling such as random digit dialing (RDD) for computer-assisted telephone interviews (CATI) and customizing interview times to maximize access in difficult-to-reach demographic groups.

### 3. Questionnaire Design

Fraym typically designs questionnaires as a consumer lifestyle and/or brand preference survey. Consumer companies frequently gather information about lifestyles, media consumption, health and wellness, and attitudes about a range of topics to establish psychographic profiles and segment the broader population into target consumer groups. Given this, the Fraym questionnaire appears almost indistinct from thousands of other ongoing consumer surveys commissioned by or on behalf of retail companies, fast moving consumer goods (FMCG) companies, or other consumer-facing businesses. As a general practice, Fraym draws heavily from research survey questions that have been successfully fielded over time. Fraym consults with regional subject matter experts (SMEs) to validate both the underlying reference survey and specific survey questions as reliable for collecting accurate and truthful responses. Moreover, Fraym consults with regional SMEs about survey module sequencing/placement and question framing considerations while constructing the questionnaire.

### 4. Survey Implementation Protocols and Controls

Fraym carefully designs the survey introduction language to accurately communicate the general focus, explicitly mention that limited respondent location data will be requested, and proactively provide privacy and anonymity assurances. These components are very important for setting respondents' expectations and building confidence in the survey itself. During survey fielding, Fraym ensures that its preferred survey vendors adhere to industry best practices. These include:

- (i) regularly testing/validating on a rolling basis to ensure participants and their responses are real and accurate
- (ii) comparing answers from respondents to pre-collected information on the same respondents for consistency, such as same age, gender, socio-economic status, and geography
- (iii) using automated natural language processing (NLP) on open-ended responses to detect nonsensical language etc.
- (iv) check for straight lining (e.g. answering "C" for all questions)
- (v) checking the speed of completion rates, (e.g. flagging anyone who spends 1/3 or less of the median time to complete the questionnaire)
- (vi) investigating interview recordings and transcripts for completeness and accuracy
- (vii) recontacting a randomly selected subset of respondents for further accuracy checks

Responses that fail any of these tests are automatically removed from the survey dataset and the survey vendor may decide to remove that respondent from their panel as well.

### 5. Response Quality Assurance

Once receiving the raw survey dataset, Fraym then applies an additional set of QA/QC measures. The following represents a subset of these data response quality assurance practices:

- **First**, Fraym uses algorithms to assess whether respondents' answers to the location-based questions (province, prefecture, county/district, postal code) are internally consistent. Next, we will convert the responses into latitude and longitude coordinates and visualize all observations in GIS software for a further human-in-the-loop assessment. Any flagged observations during these steps are excluded from the training dataset.
- **Second**, Fraym examines respondent behaviors on specific survey questions that may be particularly important or sensitive and benchmark these related metrics to other survey results as appropriate/possible.
- **Third**, Fraym explores the distribution of the surveyed data to identify any observation outliers or variables with anomalous results. Fraym data scientists then closely examine any associated observations or variables and determine whether they should be excluded from the dataset.

## Data Cleansing and Harmonization

Fraym data scientists apply a rigorous set of measures to clean, harmonize, and validate microdata inputs. The following represents a subset of these practices applied to the geo-referenced microdata:

1. **Location Validation**: Fraym uses algorithms to assess whether respondent's answers to the location-based questions (i.e., county, sub-county, ward, postal code, etc.) are internally consistent. This ensures that our survey recruitment methods consistently target the selected enumeration areas.

2. **Behavioral Benchmarking**: Fraym examines respondent behaviors on specific, particularly important or sensitive survey questions and benchmark these related metrics to other survey results as appropriate/possible. This includes respondent drop-off rates for these specific survey questions and the amount of time taken to complete each survey question when available.

3. **Outlier Detection**: We explore the distribution of the surveyed data to identify any observation outliers or variables with anomalous results. Fraym data scientists closely examine any associated observations or variables and determine whether they should be excluded from the dataset.

4. **Non-response Bias**: We pay special attention to non-response bias by question to ensure that each question has an adequate sample size and any non-response can be considered random. Missing values are treated in appropriate, industry-standard ways.

5. **Post-hoc Weighting**: After data collection, post-hoc sampling weights are created to account for any over/under sampling and ensure representativeness. Additional quality assurance is spent on the sample weights. At times, Fraym data scientists make modest adjustments to design weights (e.g., post-hoc weight adjustments), to account for any oversampling/undersampling and to ensure survey representativeness. Population parameters are drawn from population censuses, other official survey datasets (pending examination and verification), and third-party sources like the United Nations (UN), thereby ensuring comparability across countries.

For every data input, a final manual quality assurance occurs after the use of automated tools. Overall, the harmonization and cleansing process ensures that FUSEfraym™ only ingests highest quality inputs to enable robust and reliable spatial data outputs and derivative zonal statistics.

## Machine Learning Model Process

### How the Software Produces Spatial Layers

To create spatial layers from survey training data, Fraym uses a model-stacking machine learning approach to predict a continuous surface of the indicator of interest at a 1 km² resolution. This methodology builds upon existing, tested methodologies for interpolation of spatial data. The FUSEfraym™ technology creates a model that identifies correlations between the scientifically sampled survey data at enumeration clusters and typically several hundred spatial covariates from the exact location. The resulting model predicts the survey data for all non-enumerated areas. A similar approach was pioneered by USAID's Demographic and Health Surveys program in 2015 and has since been significantly improved upon by Fraym and others.

### Model-Stacking Approach

FUSEfraym's machine learning process involves generating predictions from a set of base-learner models and using those predictions to train a super-learner model. By leveraging multiple base models, the technology can improve final predictions across large geographies. Models are tuned and evaluated using industry-standard cross-validation techniques, and the predictive power of smaller data sets is increased through systems of boosting, bagging, and k-fold cross validation.

### Prediction Process

Populated grid cells with no survey data are predicted by applying a model using the parameters generated in the train and tune process. For every data layer, Fraym data scientists examine the standard model metrics such as R-squared and Root Mean Square Error (RMSE) to relay quality. Generally, data layers have very robust quality metrics. For example, a RMSE value of 0.025 for a proportional question from the survey (e.g., proportion of adults with secondary education) means that roughly the average error between the prediction and the held-out enumeration area data was 2.5 percentage points. For proportional variables, if RMSE is greater than 0.1, then data layers are not used in production. Similar thresholds are applied to non-proportional variables.

### Validation

In addition, Fraym data scientists compare the spatial surface to the lowest representative administrative level (e.g., regions) of the survey. At this level, the survey mean is compared against the implied mean of the surface when all grids are appropriately aggregated through population weighted zonal statistics. Fraym data scientists assesses whether the survey results are statistically consistent with the model-predicted mean using a binomial test. If the observed survey responses would be improbable under the model's predicted proportion, the data layer does not pass validation and is noted in the data package's documentation.

### Prediction Intervals

For each layer, Fraym also generates a pixel-level prediction interval. Prediction intervals are an estimated range of individual values that could be observed in future samples. They are used to evaluate certainty about the predictions created from a model. Fraym has adjusted existing spatial methodology to generate intervals using a combination of bootstrapping and k-fold cross validation. Our multi-step method combines a distribution of sample error terms with standard deviation to capture error and instability from multiple model types. The outputs from this approach are upper and lower bound estimates at the 1 km² level.

## Interpreting Fraym Data

Fraym model outputs are aggregated to different geographic levels (e.g., neighborhoods, cities, counties, provinces, etc.,) as needed. In Kenya, this includes counties, sub-counties, and wards. In Nigeria, this includes states, local government areas, and wards. In addition, these layers can be combined with other information, such as population densities, city boundaries, and points of interest to provide additional insights to inform decision making.

Proportions can be interpreted as the proportion of the population displaying a given characteristic. For example, if in our data extract one witnesses a 20 percent use of contraceptives among adolescents and young adults in a given ward, that should be interpreted as indicating that 20 percent of adolescents and young adults (ages 15-24) living in that ward use contraceptives.

## Time Series Comparability

Fraym provides information on a variety of sociodemographic indicators. Some of these indicators, such as media usage and attitudes, can be tracked over time. Fraym has taken steps to ensure that data layers on these indicators can be compared over time. Starting with the inputs, survey data are balance tested using statistical measures to ensure comparability between quota and survey and between survey periods. This ensures that changes in the input data are due to factors exogenous to population differences.

Our spatial covariates are aggregated to capture time specific correlations based on spatiotemporal patterns in the model covariates. Models are then tuned and tested on comparable input data and covariates across time periods. Fraym also adjusts the modeling process by reducing the number of run super learners in order to maintain model stability and performance across time.

By stabilizing the inputs data and processes, data layers produced with the same indicator of interest can be compared over time.

## Validation Against Ground Truth

Fraym has performed a quasi-ground truth validation exercise in a number of countries. Fraym first identified potential test countries for census validation. Ideally, this country would have an accessible census conducted within one to two years of a high-quality household survey. Two census candidates were selected: Tanzania 2012 and Rwanda 2012.

Because the number of individuals with some characteristic (e.g. number of people with secondary education) is often of importance to users, Fraym converted the modeled proportion surface to population totals by multiplying with a standard population raster - in this case LandScan 2012. One inconsistency addressed was the difference between LandScan and Census population totals. For example, the Tanzania 2012 Census has a total population of approximately 44 million and the 2012 LandScan raster has a total population of about 46 million (for Tanzania). To compare the modeled surfaces and Census equitably, Fraym used a rate difference metric.

The rate difference is calculated for each administrative division by dividing the population identified by the indicator by the total population of that administrative division (i.e., the number of people who have completed secondary education or higher, divided by the total number of people in that administrative division). For the modeled surface, the denominator (total population) is summed from LandScan. The Census denominator (total population) is calculated from the Census survey data. The rate difference, as calculated by Fraym, is the absolute difference between these two outputs.

The most granular geographic unit in the Tanzania Census is the district, or the second administrative boundary level. Consequently, Fraym's modeled raster data was compared to the census data by calculating the mean rate difference at the district level. For the Census, the district variable was used to calculate rate differences and for the Fraym modeled raster the rate difference was calculated using Fraym's internal weighted zonal statistics methodology. In Rwanda, the most granular administrative unit in the census was the sector, or the third administrative division. The same process noted above was used for Rwanda.

Fraym identified a set of variables for which data was available in both the high-quality household survey and the census. Absolute differences at the lowest level available in the census are presented for these variables averaged at the national level. For example, in Jenda district for secondary education the absolute rate difference is 0.0001277. The census mean for secondary education is 2.863 percent and the aggregated predictions from the Fraym model calculate the mean to be 2.851 percent, less than a 1 percentage point difference. The average of these percentage point differences in the benchmark indicator values for all districts or sectors is presented below.

**ABSOLUTE RATE DIFFERENCE FOR BENCHMARK INDICATORS, TANZANIA AND RWANDA:**

Tanzania District Average:
- Upper- and Middle-Class Consumers: 1.0%
- Secondary Education: 2.9%

Rwanda Sector Average:
- Piped Water: 2.9%
- Upper- and Middle-Class Consumers: 1.6%
- Asset Ownership: 1.5%
- Secondary Education: 1.3%

## Standard Geospatially Derived Model Covariates

1. Distance to electrical grid
2. Accessibility to cities
3. Built up surface area
4. GDP economic activity
5. Global impervious surface
6. Distance to primary road
7. Global poverty estimate
8. Distance to primary roads
9. GPW population density (2015)
10. GPW population density estimates (2020)
11. Political boundaries (i.e., states or provinces)

MODIS Products (Summary Statistics):
12. Fpar
13. Leaf area index
14. Surface reflectance bands 1-7
15. Land surface temperature
16. EVI
17. NDVI

Soil Grids:
18. Soil Ph Levels
19. Sand content
20. Carbon content

Nighttime Lights:
21. Nighttime Lights

Precipitation:
22. Annual precipitation
23. Precipitation – wettest month
24. Precipitation – driest month
25. Annual mean temperature
26. Max Temperature – warmest month
27. Min Temperature – coldest month

Population:
28. WorldPop – population count

Additional Fraym Created Features:
- Elevation derivatives
- Transformations of above features
- Demographic and socioeconomic covariates derived from national population and housing censuses, or other large official household survey results
